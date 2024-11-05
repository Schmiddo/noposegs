#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.general_utils import pose_error, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, anisotropy_loss
from utils.opt_utils import test_time_opt

from torch.utils.tensorboard import SummaryWriter

PROGRESS_BAR_UPDATE_ITERS = 50


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: list[int],
    saving_iterations: list[int],
    checkpoint_iterations: list[int],
    checkpoint: str,
    debug_from: int,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset)
    gaussians.create_from_pcd(scene.point_cloud, scene.cameras_extent)
    gaussians.training_setup(opt)
    scene.training_setup(opt)

    if network_gui is not None:
        network_gui.set_initial_pose(scene.train_cameras[0].gt_pose.matrix())

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    cam_losses = torch.zeros(len(scene.train_cameras))
    cam_rot_errors = torch.zeros_like(cam_losses)
    cam_pos_errors = torch.zeros_like(cam_losses)
    losses = torch.zeros(len(scene.train_cameras))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui is not None:
            network_gui.set_gaussians(gaussians)

        iter_start.record()

        if args.coarse_to_fine:
            # start with low-res images; double resolution every 1000 steps
            scene.set_camera_scale(max(2 ** (5 - (iteration // 1000)), 1))
        gaussians.update_learning_rate(iteration)
        scene.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # start with SH only after we reached the final resolution
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = [
                c for c in scene.getTrainCameras()
            ]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.image_with_background(bg)
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

        cam_losses[viewpoint_cam.uid] = loss.detach()
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()

        iter_end.record()

        with torch.no_grad():
            rot_error, pos_error = pose_error(
                viewpoint_cam.gt_pose.matrix(), viewpoint_cam.pose.matrix()
            )
            cam_rot_errors[viewpoint_cam.uid] = rot_error
            cam_pos_errors[viewpoint_cam.uid] = pos_error
            if not viewpoint_stack and tb_writer is not None:
                for i, l in enumerate(cam_losses):
                    tb_writer.add_scalar(f"cam_losses/cam{i}", l, iteration)
                    tb_writer.add_scalar(
                        f"cam_rot_errors/cam{i}", cam_rot_errors[i], iteration
                    )
                    tb_writer.add_scalar(
                        f"cam_pos_errors/cam{i}", cam_pos_errors[i], iteration
                    )
            elapsed = 0
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            # Progress bar
            if iteration % PROGRESS_BAR_UPDATE_ITERS == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log.item():.{7}f}"})
                progress_bar.update(PROGRESS_BAR_UPDATE_ITERS)
                elapsed = iter_start.elapsed_time(iter_end)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                elapsed,
                testing_iterations,
                scene,
                gaussians,
                render,
                pipe,
                background,
                opt.test_time_opt_steps,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                gaussians.save_iteration(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
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
                    torch.cuda.nvtx.range_push("densify and prune")
                    if opt.num_points_limit > 0:
                        with torch.no_grad():
                            max_opacity = gaussians.get_opacity.max()
                            to_prune = max(gaussians._xyz.shape[0] - opt.num_points_limit, 0)
                            thresholds = torch.linspace(0.005, 0.9 * max_opacity, 100, device="cuda")
                            opacity_threshold = ((gaussians.get_opacity < thresholds).sum(0) - to_prune).abs().argmin()
                            opacity_threshold = thresholds[opacity_threshold]
                    else:
                        opacity_threshold = 0.005
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
                if scene.optimizer is not None:
                    torch.cuda.nvtx.range_push("Camera optimizer step")
                    scene.optimizer.step()
                    viewpoint_cam.update_pose()
                    scene.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.nvtx.range_pop()

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path / f"/chkpnt{iteration}.pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if args.disable_logging:
        print("Logging disabled")
    else:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(
    tb_writer,
    iteration: int,
    Ll1: torch.Tensor,
    loss: torch.Tensor,
    l1_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elapsed: float,
    testing_iterations: list[int],
    scene: Scene,
    gaussians: GaussianModel,
    renderFunc,
    pipe,
    background: torch.Tensor,
    tto_steps,
):
    if tb_writer and iteration % 30 == 0:
        gt_poses, pred_poses = scene.gt_poses().detach(), scene.pred_poses().detach()
        rot_error, pos_error = pose_error(gt_poses, pred_poses, True)
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.detach(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.detach(), iteration)
        if elapsed > 0:
            tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "pose_error/rotation", rot_error.mean().detach(), iteration
        )
        tb_writer.add_scalar(
            "pose_error/position", pos_error.mean().detach(), iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        def _subsample(cams: list[Camera]):
            return [cams[idx % len(cams)] for idx in range(5, 30, 5)]

        validation_configs = (
            # TODO: align cameras
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": _subsample(scene.getTrainCameras())},
        )

        def _log_view(cfg_name: str, cam: Camera, img, slug: str):
            tb_writer.add_image(
                f"{cfg_name}_view_{cam.image_name}/{slug}",
                img,
                global_step=iteration,
            )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                cams: list[Camera] = config["cameras"]
                for idx, viewpoint in enumerate(cams):
                    opt_view = test_time_opt(
                        viewpoint, gaussians, pipe, background, steps=tto_steps
                    )
                    image = renderFunc(opt_view, gaussians, pipe, background)
                    image = torch.clamp(image["render"], 0, 1)
                    gt_image = opt_view.image_with_background(background)
                    gt_image: torch.Tensor = torch.clamp(gt_image, 0, 1)

                    if tb_writer and (idx < 5):
                        _log_view(config["name"], opt_view, image, "render")
                        if iteration == testing_iterations[0]:
                            _log_view(
                                config["name"], opt_view, gt_image, "ground_truth"
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                num_pts = gaussians._xyz.shape[0]
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']:>5}:"
                    f" L1 {l1_test:.5f} PSNR {psnr_test:>2.5f} #points {num_pts:>6}"
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams.add(parser)
    op = OptimizationParams.add(parser)
    pp = PipelineParams.add(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    from gaussian_renderer.network_gui import init_network_gui
    network_gui = init_network_gui(source_path=args.source_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
