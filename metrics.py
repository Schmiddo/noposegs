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

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from lpipsPyTorch.modules.lpips import LPIPS
from utils.general_utils import compute_traj_metrics, read_poses, spherify_poses, pose_error, align_poses
from utils.image_utils import psnr
from utils.loss_utils import ssim

from dataclasses import dataclass

_lpips = None
QUIET = False


# scalings as used by BARF
scales = {
    "fern": 0.0785,
    "flower": 0.0591,
    "fortress": 0.1046,
    "horns": 0.1203,
    "leaves": 0.0379,
    "orchids": 0.0920,
    "room": 0.1245,
    "trex": 0.0913,
}


def lpips(x, y):
    global _lpips
    if _lpips is None:
        _lpips = LPIPS(net_type="vgg").cuda()
    return _lpips(x, y)


def readImages(renders_dir, gt_dir):
    def read_image(fname):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        return tf.to_tensor(render)[:3], tf.to_tensor(gt)[:3], fname

    images = thread_map(read_image, os.listdir(renders_dir), disable=False)

    renders, gts, fnames = zip(*images)
    renders = torch.stack(renders).cuda()
    gts = torch.stack(gts).cuda()
    return renders, gts, fnames


def read_pred_poses(pose_path):
    gt_pose, pose = read_poses(pose_path, "cpu")
    return gt_pose, pose


@dataclass
class Result:
    PSNR: torch.Tensor
    SSIM: torch.Tensor
    LPIPS: torch.Tensor
    RPE_t: torch.Tensor
    RPE_r: torch.Tensor
    ATE: torch.Tensor
    ROT: torch.Tensor
    POS: torch.Tensor

    @staticmethod
    def calculate(renders, gts, pred_poses=None, gt_poses=None):
        if pred_poses is not None:
            rpe_r, rpe_t, ate = compute_traj_metrics(pred_poses, gt_poses)
            pred_poses = align_poses(pred_poses, gt_poses)
            abs_rot_err, abs_pos_err = pose_error(pred_poses, gt_poses)
        else:
            rpe_r = rpe_t = ate = abs_rot_err = abs_pos_err = torch.zeros(1)

        ssims = []
        psnrs = []
        lpipss = []
        frames = tqdm(
            zip(renders, gts),
            total=len(renders),
            desc="Metric evaluation progress",
            disable=QUIET,
        )
        for render, gt in frames:
            ssims.append(ssim(render, gt))
            psnrs.append(psnr(render, gt))
            lpipss.append(lpips(render, gt))

        return Result(
            torch.stack(psnrs),
            torch.stack(ssims),
            torch.stack(lpipss),
            rpe_t,
            rpe_r,
            ate,
            abs_rot_err,
            abs_pos_err,
        )

    @property
    def metrics(self):
        return {k: v.mean().item() for k, v in self.__dict__.items()}

    def per_frame_metrics(self, image_names):
        # structure is {metric1: {frame1: value1, frame2: value2, ...}, ...}
        return {
            k: {n: v for n, v in zip(image_names, v.tolist())}
            for k, v in self.__dict__.items() if k in ("SSIM", "PSNR", "LPIPS")
        }


def evaluate(model_paths, spherify, global_scale=None):
    full_dict = {}
    per_view_dict = {}

    for scene_dir in model_paths:
        scale = global_scale or scales.get(Path(scene_dir).name, 1.0)
        if not QUIET: print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"
        if not test_dir.exists():
            print(f"WARNING: {test_dir} does not exist, skipping")
            continue

        for method in os.listdir(test_dir):
            if not QUIET: print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            # hacky
            iteration = method.split("_")[1]
            pose_path = Path(scene_dir + f"/train_cameras_{iteration}.pth")
            if pose_path.exists():
                gt_poses, pred_poses = read_pred_poses(pose_path)
                gt_poses[:, :3, 3] *= scale
                if spherify:
                    gt_poses = spherify_poses(gt_poses)
            else:
                gt_poses = pred_poses = None

            result = Result.calculate(renders, gts, pred_poses, gt_poses)
            if not QUIET:
                for name, value in result.metrics.items():
                    print(f"  {name:5}: {value:>12.7f}")
            
            full_dict[scene_dir][method].update(result.metrics)
            per_view_dict[scene_dir][method].update(result.per_frame_metrics(image_names))

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

    return full_dict


def print_table(full_dict, format="plain"):
    formats = {
        "plain": (" ", ""),
        "latex": (" & ", "\\\\"),
    }
    sep, eol = formats[format]
    for scene_dir in full_dict:
        scene = Path(scene_dir).name
        for method in full_dict[scene_dir]:
            line = sep.join([f"{v:>10.3f}" for v in full_dict[scene_dir][method].values()])
            print(f"{method:15}{sep}{scene:15}{sep}{line} {eol}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--spherify_poses", action="store_true")
    parser.add_argument("--pose_scale", type=float, default=None)
    parser.add_argument("--print_table", action="store_true")
    parser.add_argument("--table_format", default="plain", choices=["plain", "latex"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    QUIET = args.quiet
    full_dict = evaluate(args.model_paths, args.spherify_poses, args.pose_scale)

    if args.print_table:
        print_table(full_dict, args.table_format)
