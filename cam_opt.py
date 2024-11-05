import torch
import torchvision.transforms.functional as TF
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from arguments import ParamGroup, ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.pose import random_pose
from utils.general_utils import pose_error, safe_state, get_lr_scheduler, euler_rotation, get_quat, procrustes, read_poses, se3_from_mat4, Scheduler
from utils.loss_utils import l1_loss, ssim, smoothness_loss

from lietorch import SE3


PROGRESS_BAR_UPDATE_ITERS = 50

import cv2
from skimage.util import random_noise
from cudaops import blur2d

NOISE_TYPES = {
    "gaussian": lambda img, v: random_noise(img, mode="gaussian", var=v*v),
    "s_and_p": lambda img, v: random_noise(img, mode="s&p", amount=v),
    "pepper": lambda img, v: random_noise(img, mode="pepper", amount=v),
    "salt": lambda img, v: random_noise(img, mode="salt", amount=v),
    "poisson": lambda img, _: random_noise(img, mode="poisson"),
}


@dataclass
class OptimizationParams(ParamGroup):
    iterations: int = 1000
    cam_lr_init: float = 1e-3
    cam_lr_final: float = 1e-5
    cam_lr_delay_steps: int = 0
    cam_lr_delay_mult: float = 0.01
    cam_lr_scheduler: str = "cosine"
    lambda_dssim: float = 0.2
    random_background: bool = False
    bruteforce_iters: int = 0

    depth_smoothness_weight: float = 0.0
    random_masking: float = 0.0

    num_cam_variations: int = 5
    use_prior: bool = False
    forward_facing: bool = False
    _description = "Optimization Parameters"


@dataclass
class NoiseParams(ParamGroup):
    noise_types: list[str] = field(default_factory=lambda: [])
    n_mask: int = 0
    r_mask: int = 0
    delta_brightness: float = 0.0
    sigma: float = 0.0
    amount: float = 0.0
    _description = "Noise Options"


def save_results(results_file, results):
    torch.save(results, results_file)


def random_pose_euler(bounds_rotation=15, bounds_translation=0.25):
    rotation = bounds_rotation * (2 * torch.rand(3) - 1)
    rotation = euler_rotation(rotation[0], rotation[1], rotation[2])
    rotation = get_quat(rotation[:3, :3])
    translation = bounds_translation * (2 * torch.rand(3) - 1)
    return torch.cat([translation, rotation[1:], rotation[:1]]).cuda()


def random_pose_iingp(initial_pose, bounds_rotation=15, bounds_translation=0.25):
    # rotate camera in local coordinates, translate in world coordinates
    from lietorch import SE3
    noise = random_pose_euler(bounds_rotation, bounds_translation)
    translation = noise.clone()
    translation[3:-1] = 0
    translation[-1] = 1
    noise[:3] = 0
    initial_trans = initial_pose.data.clone()
    initial_trans[3:] = 0
    initial_trans[-1] = 1
    initial_rot = initial_pose.data.clone()
    initial_rot[:3] = 0
    noised_pose = SE3(translation) * SE3(initial_trans) * SE3(noise) * SE3(initial_rot)
    return noised_pose.data


def disturb_camera_image(cam: Camera, args: NoiseParams):
    img, alpha = cam.original_image, cam.original_alpha_mask
    if alpha is None:
        alpha = torch.ones_like(img[:1])
    img, alpha = add_image_noise(
        img,
        alpha,
        args.noise_types,
        args.sigma,
        args.amount,
        args.delta_brightness,
        args.n_mask,
        args.r_mask,
    )
    cam.original_image, cam.original_alpha_mask = img, alpha


def add_image_noise(img, alpha, noise_types=[], sigma=0.0, amount=0.0, delta_brightness=0.0, n_mask=0, r_mask=0):
    # adapted from https://github.com/NVlabs/ParallelInversion
    img = img.permute(1, 2, 0).cpu().numpy()
    alpha = alpha[0].cpu().numpy()

    if delta_brightness != 0:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bimg = img[..., 2]
        if delta_brightness < 0:
            bimg[bimg < abs(delta_brightness)] = 0
            bimg[bimg >= abs(delta_brightness)] += delta_brightness
        else:
            lim = 1 - delta_brightness
            bimg[bimg > lim] = 1
            bimg[bimg <= lim] += delta_brightness
        img[..., 2] = bimg
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    for noise in noise_types:
        img = NOISE_TYPES[noise](img, sigma if noise == "gaussian" else amount)
        img = np.clip(img, 0, 1)

    if r_mask > 0 and n_mask > 0:
        kernel = np.ones((10, 10))
        gradient = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)
        yy, xx = np.where(gradient >= 0.9 * np.max(gradient))

        mask_combined = np.ones(alpha.shape, np.uint8) * 255
        for i in range(n_mask):
            idx = np.random.choice(len(xx))
            mask = cv2.circle((alpha * 255).astype(np.uint8), (xx[idx], yy[idx]), r_mask, (0, 0, 0), -1)
            mask_combined = cv2.bitwise_and(mask_combined, mask)

        alpha = mask_combined[None, ...].astype(np.float32) / 255

    return (
        torch.from_numpy(img).float().cuda().permute(2, 0, 1),
        torch.from_numpy(alpha).float().cuda(),
    )

    
def optimize(trained_scene: Path, scene_args: ModelParams, optim_args: OptimizationParams, pipe_args: PipelineParams, noise_args: NoiseParams):
    # pipe_args.debug = True
    model_path = Path(scene_args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(scene_args.sh_degree)
    scene = Scene(scene_args)

    gaussians.load_ply(trained_scene)
    gaussians.active_sh_degree = scene_args.sh_degree
    print(f"Model has {gaussians._xyz.shape[0]} points")

    bg_color = [1, 1, 1] if scene_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    rot_errors, pos_errors = [], []
    results_per_step, poses_per_step = {}, {}
    for i, testcam in enumerate(scene.test_cameras):
        if len(scene.test_cameras) > 70 and (i % 7 != 0 or i % 5 != 0):
            continue
        print(f"Cam {i: 2}")
        disturb_camera_image(testcam, noise_args)
        for j in range(optim_args.num_cam_variations):
            cam = testcam.clone()
            bounds_rotation = 15
            bounds_translation = 0.15 if optim_args.forward_facing else 0.25
            cam._pose.data.copy_(random_pose_iingp(cam._pose, bounds_rotation, bounds_translation))
            vis_path = model_path / f"cam{i}_{j}"
            torch.cuda.nvtx.range_push("Optimization")
            rot_error, pos_error, errors_per_step, poses = optimize_cam(cam, gaussians, background, optim_args, pipe_args, vis_path)
            torch.cuda.nvtx.range_pop()
            results_per_step[f"cam{i}_{j}"] = errors_per_step.cpu()
            poses_per_step[f"cam{i}_{j}"] = poses.cpu()
            rot_errors.append(rot_error)
            pos_errors.append(pos_error)
    
    save_results(model_path / "pose_errors.pth", results_per_step)
    save_results(model_path / "poses.pth", poses_per_step)
    with open(model_path / "pose_errors.json", "w") as f:
        json_results_per_step = {c: v.tolist() for c, v in results_per_step.items()}
        json.dump(json_results_per_step, f)
    
    rot_errors = torch.stack(rot_errors)
    pos_errors = torch.stack(pos_errors)

    rot_mean_error, pos_mean_error = rot_errors.mean(), pos_errors.mean()
    rot_accuracy, pos_accuracy = (rot_errors < 5).float().mean(), (pos_errors < 0.05).float().mean()
    rot_accuracy_high, pos_accuracy_high = (rot_errors < 0.1).float().mean(), (pos_errors < 0.01).float().mean()
    print("#" * 5, f"Results for {scene_args.source_path}:")
    print(f"Mean error (rot/trans):   {rot_mean_error.item():.5f} | {pos_mean_error.item():.5f}")
    print(f"Below threshold 5.0/0.05: {rot_accuracy:.5f} | {pos_accuracy:.5f}")
    print(f"Below threshold 0.1/0.01: {rot_accuracy_high:.5f} | {pos_accuracy_high:.5f}")

    with (model_path / "results.json").open("w") as f:
        json.dump({
            "ROT": rot_mean_error.item(),
            "POS": pos_mean_error.item(),
            "rot@5": rot_accuracy.item(),
            "pos@0.05": pos_accuracy.item(),
            "rot@0.1": rot_accuracy_high.item(),
            "pos@0.01": pos_accuracy_high.item(),
        }, f)


def optimize_cam(
    cam: Camera,
    gaussians: GaussianModel,
    background: torch.Tensor,
    optim_args: OptimizationParams,
    pipe_args: PipelineParams,
    vis_path: Path | None,
):
    optimizer = torch.optim.Adam(cam.parameters(), lr=optim_args.cam_lr_init, eps=1e-15)
    scheduler = get_lr_scheduler(
        optim_args.cam_lr_scheduler,
        lr_init=optim_args.cam_lr_init,
        lr_final=optim_args.cam_lr_final,
        lr_delay_steps=optim_args.cam_lr_delay_steps,
        lr_delay_mult=optim_args.cam_lr_delay_mult,
        max_steps=optim_args.iterations,
    )
    scheduler = Scheduler(optimizer, scheduler)

    with torch.no_grad():
        rot_error, pos_error = pose_error(cam.pose.matrix(), cam.gt_pose.matrix())
    print(f"Initial pose error (rot|trans): {rot_error.item():.5f} | {pos_error.item():.5f}")

    if vis_path is not None:
        vis_path.mkdir(exist_ok=True)
        iio.imwrite(vis_path / "target.png", cam.image_with_background(background).cpu().permute(1, 2, 0).mul(255).byte().numpy())
        with open(vis_path / "target_cam.json", "w") as f:
            camdict = {
                "K": cam.K,
                "img_shape": cam.image.shape[-2:],
                "pose": cam.gt_pose.matrix().cpu().tolist(),
            }
            json.dump(camdict, f)

    progress_bar = tqdm(range(optim_args.iterations), desc="Steps")
    steps, poses = [], []
    start = [torch.cuda.Event(enable_timing=True) for _ in range(optim_args.iterations)]
    stop = [torch.cuda.Event(enable_timing=True) for _ in range(optim_args.iterations)]
    for iteration in range(optim_args.iterations):
        start[iteration].record()
        torch.cuda.nvtx.range_push("Rendering")
        bg = torch.rand(3, device="cuda") if optim_args.random_background else background
        pkg = render(cam, gaussians, pipe_args, bg)
        img, depth = pkg["render"], pkg["depth"]
        gt_image = cam.image_with_background(bg)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Loss computation")
        if optim_args.random_masking > torch.rand(()):
            H, W = gt_image.shape[-2:]
            mh, mw = H//2, W//2
            mask = torch.ones_like(gt_image)
            # t = torch.randint(0, H - mh, ())
            # l = torch.randint(0, W - mw, ())
            r = torch.rand(())
            if r < 0.25:
                t = l = 0
            elif r < 0.5:
                t = 0; l = W//2
            elif r < 0.75:
                t = H//2; l = 0
            else:
                t = H//2; l = W//2
            mask[:, t:t+mh, l:l+mw] = 0
            gt_image = mask * gt_image
            img = mask * img
        if optim_args.use_prior:
            def _project(p):
                return p[..., :2] / p[..., -1:]
            loss = (0.995**iteration) * _project(cam.projection_matrix[:3, :3] @ (cam.pose.inv() * gaussians.get_xyz.mean(dim=0))).abs().sum().sub(0.25).clamp(0)
        else:
            loss = 0
        loss += (
            (1 - optim_args.lambda_dssim) * l1_loss(img, gt_image)
            + optim_args.lambda_dssim * (1 - ssim(img, gt_image))
        )
        if optim_args.depth_smoothness_weight > 0:
            loss += optim_args.depth_smoothness_weight * smoothness_loss(depth, gt_image)
        torch.cuda.nvtx.range_pop()

        loss.backward()

        torch.cuda.nvtx.range_push("Optimizer and scheduler")
        optimizer.step()
        scheduler.step(iteration)
        cam.update_pose()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.nvtx.range_pop()
        stop[iteration].record()
        
        with torch.no_grad():
            rot_error, pos_error = pose_error(cam.pose.matrix(), cam.gt_pose.matrix())

            steps.append(torch.stack([loss, rot_error, pos_error]))
            poses.append(cam.pose.data)

            if iteration % PROGRESS_BAR_UPDATE_ITERS == 0:
                pbar_dict = {
                    "Loss": f"{loss.item():.5f}",
                    "dRot": f"{rot_error.item():.5f}",
                    "dTrans": f"{pos_error.item():.5f}",
                }
                progress_bar.set_postfix(pbar_dict)
                progress_bar.update(PROGRESS_BAR_UPDATE_ITERS)

    progress_bar.close()
    steps = torch.stack(steps)
    poses = torch.stack(poses)

    with torch.no_grad():
        rot_error, pos_error = pose_error(cam.pose.matrix(), cam.gt_pose.matrix())
    stop[-1].synchronize()
    elapsed = torch.as_tensor([a.elapsed_time(b) for a, b in zip(start, stop)])
    print(f"Final pose error (rot|trans): {rot_error.item():.5f} | {pos_error.item():.5f} in {elapsed.sum()/1000:.2f}s")
    steps = torch.cat([elapsed[:, None], steps.cpu()], dim=-1)
    return rot_error.detach(), pos_error.detach(), steps, poses


if __name__ == "__main__":
    parser = ArgumentParser()
    scene_args = ModelParams.add(parser)
    optim_args = OptimizationParams.add(parser)
    pipe_args = PipelineParams.add(parser)
    noise_args = NoiseParams.add(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    args.eval = True

    print(f"Running camera optimization for {args.model_path}")

    safe_state(args.quiet)

    optimize(
        args.checkpoint,
        scene_args.extract(args),
        optim_args.extract(args),
        pipe_args.extract(args),
        noise_args.extract(args),
    )
