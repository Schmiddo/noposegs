#
# Based on code from "3D Gaussian Splatting for Real-Time Radiance Field Rendering" by Kerbl et al.
# https://github.com/graphdeco-inria/gaussian-splatting
#

import os
import sys
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel, GaussianPointCloud, get_pointcloud_optimizer
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import compute_traj_metrics, safe_state, unproject_points, get_cosine_lr_func, Scheduler, align_poses
from utils.loss_utils import l1_loss, ssim, anisotropy_loss

from torch.utils.tensorboard.writer import SummaryWriter

from lietorch import SE3


PROGRESS_BAR_UPDATE_ITERS = 50
network_gui = None
logger = None


@torch.no_grad()
def model_from_cams(opt: OptimizationParams, initial_cams: list[Camera], pts_per_cam, sh_degree=0):
    total_points = []
    total_colors = []
    for initial_cam in initial_cams:
        mask = initial_cam.alpha_mask
        if mask is not None:
            mask.squeeze_()
        depth = initial_cam.depth * 1/0.1378
        initial_points = unproject_points(depth, initial_cam.projection_matrix, mask)
        initial_colors = initial_cam.image
        if mask is not None:
            initial_colors = initial_colors[..., mask > 0]
        initial_colors = initial_colors.flatten(1)
        N = initial_points.shape[-1]
        px_idcs = torch.argsort(torch.randn(N))[:pts_per_cam]
        # transform into camera coordinate system
        initial_points = initial_cam.cam2world[:3, :3] @ initial_points + initial_cam.cam2world[:3, 3][:, None]
        total_points.append(initial_points[:, px_idcs].transpose(-1, -2))
        total_colors.append(initial_colors[:, px_idcs].transpose(-1, -2))
    total_points = torch.cat(total_points, dim=0)
    total_colors = torch.cat(total_colors, dim=0)

    gaussians = GaussianPointCloud.from_pointcloud(total_points, total_colors)
    optimizer = get_pointcloud_optimizer(opt, gaussians)
    return gaussians, optimizer


def optimize(
        desc, nsteps, cams: list[Camera], gaussians: GaussianModel,
        optimizers: list[torch.optim.Optimizer], pipe, background, schedulers=[],
        mask_loss=False, lambda_dssim=0.2,
    ):
    losses = []
    progress_bar = tqdm(
        range(nsteps),
        leave=False,
        total=nsteps,
        disable=nsteps < PROGRESS_BAR_UPDATE_ITERS,
        desc=desc,
    )
    for i in range(nsteps):
        viewpoint_cam = cams[torch.randint(len(cams), (1,))]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if network_gui is not None:
            network_gui.set_gaussians(gaussians)

        # Loss
        gt_image = viewpoint_cam.image_with_background(background)
        if mask_loss:
            with torch.no_grad():
                mask = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    torch.zeros(3, device="cuda"),
                    override_color=torch.ones((len(gaussians), 3), device="cuda")
                )["render"]
                mask = (mask > 0.99).float()
        else:
            mask = None
        Ll1 = l1_loss(image, gt_image, mask)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        if i % PROGRESS_BAR_UPDATE_ITERS == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
            progress_bar.update(PROGRESS_BAR_UPDATE_ITERS)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step(i)
        losses.append(loss.detach())
        with torch.no_grad():
            if hasattr(gaussians, "add_densification_stats"):
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)
            # Apply pose gradient to pose
            viewpoint_cam.update_pose()
    return torch.stack(losses)


def print_traj_metrics(pred_poses, gt_poses):
    rel_rot_err, rel_pos_err, traj_err = compute_traj_metrics(pred_poses, gt_poses)

    print(f"RPE_r: {rel_rot_err.abs().mean().item():.4f}")
    print(f"RPE_t: {rel_pos_err.mean().item():.4f}")
    print(f"ATE  : {traj_err.item():.4f}")


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
):
    scene = Scene(dataset)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if network_gui is not None:
        # network_gui.set_initial_pose(scene.train_cameras[0].gt_pose.inv().matrix())
        network_gui.set_intrinsics(scene.train_cameras[0].K)

    scene.training_setup(opt)
    cam_optimizer = scene.optimizer
    cam_scheduler = Scheduler(cam_optimizer, scene.scheduler)
    progress_bar = tqdm(range(len(scene.train_cameras)))

    pointclouds = []
    abs_poses = [torch.as_tensor(SE3.id_elem).cuda()]
    rel_poses = []
    edges = []

    if opt.num_previous_frames > 0:
        for i in progress_bar:
            c1 = scene.train_cameras[i]
            with torch.no_grad():
                points, point_optimizer = model_from_cams(opt, [c1], opt.points_per_frame)
                pointclouds.append(points)
            optimize("New frame", opt.per_frame_point_steps, [c1], points, [point_optimizer], pipe, background)

            # fit previous cam(s) to new point cloud to initialize pose
            for j in range(opt.num_previous_frames):
                if j >= i: break
                c2 = scene.train_cameras[i].clone()
                cam_optimizer = torch.optim.Adam(c2.parameters(), lr=0.0, eps=1e-15)
                schedule_fn = get_cosine_lr_func(opt.per_frame_cam_lr_init, opt.cam_lr_final, max_steps=opt.per_frame_camopt_steps)
                cam_scheduler = Scheduler(cam_optimizer, schedule_fn)
                points = pointclouds[i - (j + 1)]
                optimize(f"Relative pose cam {i-j-1}", opt.per_frame_camopt_steps, [c2], points, [cam_optimizer], pipe, background, [cam_scheduler], mask_loss=True)
                with torch.no_grad():
                    if j == 0:
                        abs_poses.append((c2.pose * SE3(abs_poses[-1])).data)
                    rel_poses.append(c2.pose.data)
                    edges.append((i - (j + 1), i))
            scene.train_cameras[i]._pose.data.copy_(abs_poses[-1])

            with torch.no_grad():
                pred_poses, gt_poses = scene.pred_poses(), scene.gt_poses()
                if i > 1:
                    rel_rot_err, rel_pos_err, traj_err = compute_traj_metrics(pred_poses[:i], gt_poses[:i])
                    pred_poses = align_poses(pred_poses[:i], gt_poses[:i])
                    pred_poses = gt_poses[0] @ torch.linalg.inv(pred_poses[0]) @ pred_poses[:i]
                    progress_bar.set_postfix({
                        "rot": f"{rel_rot_err.abs().mean().item():.4f}",
                        "pos": f"{rel_pos_err.mean().item():.4f}",
                        "ate": f"{traj_err.item():.4f}",
                    })

        with torch.no_grad():
            print("After online phase")
            print_traj_metrics(scene.pred_poses(), scene.gt_poses())

    with torch.no_grad():
        # scene optimization
        if opt.offline_point_init == "unproject" and dataset.load_depth:
            from utils.sh_utils import SH2RGB
            pcd, _ = model_from_cams(opt, scene.train_cameras, 500)
            points = pcd._xyz.detach()
            colors = SH2RGB(pcd._features_dc.detach()).squeeze()
        elif opt.offline_point_init == "random":
            poses = scene.pred_poses()[:, :3, 3]
            print("max", poses.max(dim=0).values, "min", poses.min(dim=0).values)
            npoints = 10
            points = 0.25 * (poses.max(dim=0).values - poses.min(dim=0).values) * (2 * torch.rand(npoints, 3, device="cuda") - 1)
            points = points + poses.mean(dim=0)
            colors = torch.rand(npoints, 3, device="cuda")
        else:
            raise ValueError(f"Unknown init method '{opt.offline_point_init}'")

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.create_from_pcd(BasicPointCloud(points, colors, None), spatial_lr_scale=1)
    gaussians.training_setup(opt)
    scene.training_setup(opt)
    viewpoint_stack = None

    if network_gui is not None:
        network_gui.set_gaussians(gaussians)

    ema_loss_for_log = 0
    progress_bar = tqdm(range(opt.iterations))
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        scene.update_learning_rate(iteration)
    
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        if not viewpoint_stack:
            viewpoint_stack = [
                c for c in scene.getTrainCameras()
            ]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.image_with_background(background)
        torch.cuda.nvtx.range_push("loss forward")
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        if opt.opacity_loss_weight > 0 and iteration < opt.opacity_loss_until_iter:
            loss += opt.opacity_loss_weight * gaussians.get_opacity.abs().mean()

        if opt.anisotropy_loss_weight > 0:
            loss += opt.anisotropy_loss_weight * anisotropy_loss(gaussians.get_scaling, opt.anisotropy_max_ratio)
        torch.cuda.nvtx.range_pop()

        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            # Progress bar
            if iteration % PROGRESS_BAR_UPDATE_ITERS == 0:
                pred_poses, gt_poses = scene.pred_poses(), scene.gt_poses()
                rel_rot_err, rel_pos_err, traj_err = compute_traj_metrics(pred_poses, gt_poses)
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log.item():.{7}f}",
                    "rot": f"{rel_rot_err.abs().mean().item():.4f}",
                    "pos": f"{rel_pos_err.mean().item():.4f}",
                    "ate": f"{traj_err.item():.4f}",
                    "N": f"{gaussians._xyz.shape[0]}",
                })
                progress_bar.update(PROGRESS_BAR_UPDATE_ITERS)
            if iteration == opt.iterations:
                progress_bar.close()
        
            # Densification
            if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
                torch.cuda.nvtx.range_push("densification stats")
                # Keep track of max radii in image-space and sum of 2D gradient norms
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, radii
                )
                torch.cuda.nvtx.range_pop()

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    if opt.num_points_limit > 0:
                        with torch.no_grad():
                            max_opacity = gaussians.get_opacity.max()
                            to_prune = max(gaussians._xyz.shape[0] - opt.num_points_limit, 0)
                            thresholds = torch.linspace(0.005, 0.9 * max_opacity, 100, device="cuda")
                            opacity_threshold = ((gaussians.get_opacity < thresholds).sum(0) - to_prune).abs().argmin()
                            opacity_threshold = thresholds[opacity_threshold]
                    else:
                        opacity_threshold = 0.005
                    torch.cuda.nvtx.range_push("densify and prune")
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    torch.cuda.nvtx.range_pop()

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        torch.cuda.nvtx.range_push("reset_opacity")
                        gaussians.reset_opacity()
                        torch.cuda.nvtx.range_pop()

            # Optimizer step
            if iteration < opt.iterations:
                torch.cuda.nvtx.range_push("Gaussian optimizer step")
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.nvtx.range_pop()
                if scene.optimizer is not None and iteration < opt.cam_lr_max_steps:
                    torch.cuda.nvtx.range_push("Camera optimizer step")
                    scene.optimizer.step()
                    viewpoint_cam.update_pose()
                    scene.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.nvtx.range_pop()

            if iteration % 100 == 0 or iteration == 1:
                scene.save(iteration)

    # training end, save model
    print(f"\n[ITER {iteration}] Saving Gaussians")
    scene.save(iteration)
    gaussians.save_iteration(scene.model_path, iteration)

    with torch.no_grad():
        print_traj_metrics(scene.pred_poses(), scene.gt_poses())


def prepare_output_and_logger(model_path: str):
    if not model_path:
        raise ValueError("Need to set a model path")

    # Set up output folder
    print(f"Output folder: {model_path}")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    return SummaryWriter(model_path)


if __name__ == "__main__":
    from gaussian_renderer.network_gui import init_network_gui
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams.add(parser)
    op = OptimizationParams.add(parser)
    pp = PipelineParams.add(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui = init_network_gui(args.source_path, args.ip, args.port)
    if not args.disable_logging:
        logger = prepare_output_and_logger(args.model_path)
    else:
        print("Logging disabled")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
    )

    # All done
    print("\nTraining complete.")
