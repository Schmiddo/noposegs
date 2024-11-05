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
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state, procrustes, se3_from_mat4, align_forward
from utils.opt_utils import test_time_opt


def render_set(
    args, model_path, name, iteration, views, gaussians, pipeline, background
):
    render_path = os.path.join(model_path, name, args.method_name.format(iteration), "renders")
    gts_path = os.path.join(model_path, name, args.method_name.format(iteration), "gt")
    diff_path = os.path.join(model_path, name, args.method_name.format(iteration), "diff")
    depth_path = os.path.join(model_path, name, args.method_name.format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(diff_path, exist_ok=True)
    if args.render_depth:
        makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        opt_view = test_time_opt(view, gaussians, pipeline, background, lr=args.test_time_opt_lr, steps=args.test_time_opt_steps)
        with torch.no_grad():
            render_pkg = render(opt_view, gaussians, pipeline, background)
            rendering, depth = render_pkg["render"], render_pkg["depth"]
            gt = view.image_with_background(background)
            diff = (rendering - gt).abs().sum(0)
            # diff = torch.stack([rendering.sum(0), torch.zeros_like(rendering.sum(0)), gt.sum(0)], dim=0)
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            )
            torchvision.utils.save_image(
                diff, os.path.join(diff_path, "{0:05d}".format(idx) + ".png")
            )
            if args.render_depth:
                depth[depth == 0] = torch.inf
                torchvision.utils.save_image(
                    1 / (depth + 1e-6), os.path.join(depth_path, "{0:05d}".format(idx) + ".png")
                )


def render_sets(
    args,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, load_iteration=iteration)
    gaussians.load_iteration(scene.model_path, scene.loaded_iter)

    print(gaussians._xyz.shape[0], "points")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(
            args,
            dataset.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
        )

    if not skip_test:
        if args.align_cameras == "procrustes":
            s, R, t = procrustes(scene.pred_poses()[:, :3, 3], scene.gt_poses()[:, :3, 3])
            for c in scene.test_cameras:
                aligned_pose = c.pose.matrix()
                aligned_pose[:3, :3] = R @ aligned_pose[:3, :3]
                aligned_pose[:3, 3] = s * R @ aligned_pose[:3, 3] + t
                c._pose.data.copy_(se3_from_mat4(aligned_pose))
        if args.align_cameras == "forward":
            pred_poses = scene.pred_poses()
            gt_poses = scene.gt_poses()
            s, R, t = align_forward(pred_poses, gt_poses)
            for c in scene.test_cameras:
                aligned_pose = c.pose.matrix()
                aligned_pose[:3, :3] = R @ aligned_pose[:3, :3]
                aligned_pose[:3, 3] = s * R @ aligned_pose[:3, 3] + t
                c._pose.data.copy_(se3_from_mat4(aligned_pose))
        if args.align_cameras == "nearest":
            for c in reversed(scene.test_cameras):
                cmin = (c.pose.matrix()[:3, 3] - scene.gt_poses()[:, :3, 3]).norm(dim=-1).argmin().item()
                c._pose.data.copy_(scene.train_cameras[cmin].pose.data)
        if args.align_cameras == "identity":
            for c in scene.test_cameras:
                c._pose.data.copy_(c._pose.id_elem.data)
        testcams = scene.getTestCameras()
        render_set(
            args,
            dataset.model_path,
            "test",
            scene.loaded_iter,
            testcams,
            gaussians,
            pipeline,
            background,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams.add(parser, fill_none=True)
    pipeline = PipelineParams.add(parser)
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--test_time_opt_steps", default=0, type=int)
    parser.add_argument("--test_time_opt_lr", default=1e-3, type=float)
    parser.add_argument("--align_cameras", choices=["identity", "nearest", "procrustes", "forward", "none"], default="none", const="procrustes", nargs="?")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--method_name", default="ours_{}")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        args,
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
