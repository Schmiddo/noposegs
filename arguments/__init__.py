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
from argparse import ArgumentParser, Namespace

from dataclasses import dataclass, fields, field, MISSING, replace
from typing_extensions import Self


class ParamGroup:
    @classmethod
    def add(cls, parser: ArgumentParser, name=None, fill_none=False):
        group = parser.add_argument_group(name or getattr(cls, "_description", None))
        for f in fields(cls):
            key, value = f.name, f.default
            if value is MISSING:
                value = f.default_factory()
            t = type(value)
            value = None if fill_none else value

            arg_names = [f"--{key}"]
            kwargs = {"default": value, "type": t}
            if t == bool:
                kwargs["action"] = "store_true"
                kwargs.pop("type")
            if t == list:
                kwargs["nargs"] = "+"
                kwargs["type"] = str
            if t == tuple:
                kwargs["nargs"] = len(value)
                kwargs["type"] = str
            if key in getattr(cls, "_shortcuts", {}):
                arg_names.append(cls._shortcuts[key])
            group.add_argument(*arg_names, **kwargs)
        return cls()

    def extract(self, args) -> Self:
        kwargs = {k: v for k, v in vars(args).items() if k in {f.name for f in fields(self)}}
        return type(self)(**kwargs)


@dataclass
class ModelParams(ParamGroup):
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: float = -1.0
    white_background: bool = False
    data_device: str = "cuda"
    eval: bool = False
    disable_logging: bool = False
    num_views: int = -1
    view_list: list[str] = field(default_factory=lambda: [])
    cam_noise: float = 0.0
    init_cam_identity: bool = False
    load_depth: str = ""
    # only relevant if no test set is specified
    # < 1 means use fraction of train set
    # > 1 means use every nth view (will be truncated to int)
    test_hold: float = 8.0

    _description = "Loading Parameters"
    _shortcuts = {
        "source_path": "-s",
        "model_path": "-m",
        "images": "-i",
        "resolution": "-r",
        "white_background": "-w",
    }

    def extract(self, args):
        g = super().extract(args)
        return replace(g, source_path=os.path.abspath(g.source_path))


@dataclass
class PipelineParams(ParamGroup):
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    _description = "Debug Parameters"


@dataclass
class OptimizationParams(ParamGroup):
    iterations: int = 30_000
    position_lr_scheduler: str = "exp"
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    cam_lr_init: float = 0.0
    cam_lr_final: float = 1e-5
    cam_lr_delay_steps: int = 0
    cam_lr_delay_mult: float = 0.01
    cam_lr_scheduler: str = "cosine"
    cam_lr_max_steps: int = 15000
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    test_time_opt_steps: int = 0
    coarse_to_fine: bool = False

    opacity_loss_weight: float = 0.0
    opacity_loss_until_iter: int = 10_000
    anisotropy_loss_weight: float = 1.0
    anisotropy_max_ratio: float = 5.0
    # options for online, pose-free training
    pointcloud_overlap_steps: int = 0
    per_frame_point_steps: int = 100
    points_per_frame: int = 50000
    per_frame_camopt_steps: int = 200
    num_previous_frames: int = 0
    per_frame_cam_lr_init: float = 1e-3
    offline_point_init: str = "unproject"

    num_points_limit: int = 0

    _description = "Optimization Parameters"


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
