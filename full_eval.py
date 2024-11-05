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

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from argparse import ArgumentParser
from time import sleep

from utils.data_utils import (
    DATASETS, SCENESETS, Scenesets, add_datasets_to_parser
)


GS_RUNDIR = Path(os.getenv("GS_RUNDIR", "./runs"))
QUIET = False
DRYRUN = False


def run_cmd(cli):
    if DRYRUN:
        print(cli)
        return 0
    else:
        return os.system(cli)


def log(msg):
    # replace this with proper logging at some point
    if not QUIET:
        print(msg)


@dataclass
class Config:
    name: str
    scenes: list[str]
    source: Path


class Dispatcher:
    def __init__(self, script: str, outdir: Path, overwrite=False):
        self._train_script = script
        self._outdir = outdir
        self._overwrite = overwrite
        
    def _run(self, script: str, scene: Path, *extra_args):
        cli_args = (
            "python",
            script,
            f"-s {scene}",
            f"-m {self._outdir / scene.name}",
         ) + extra_args

        cmd_line = " ".join(cli_args)
        ret = run_cmd(cmd_line)
        if ret != 0:
            log(f"{scene.name}: failed!")
            sleep(2)
    
    def run_training(self, scene: Path, extra_args):
        model_path = self._outdir / scene.name
        if model_path.exists() and not self._overwrite:
            log(f"Skip training for {model_path}")
            return
        self._run(self._train_script, scene, *extra_args)
    
    def run_rendering(self, scene: Path, iteration, extra_args):
        render_path = self._outdir / scene.name / f"test/ours_{iteration}/"
        if render_path.exists() and not self._overwrite:
            log(f"Skip rendering for {render_path}")
            return
        # model_path = self._outdir / scene.name / f"point_cloud/iteration_{iteration}"
        # if not (model_path / "point_cloud.ply").exists():
        #     log(f"Skip rendering for {render_path}: no point cloud found")
        #     return
        self._run("render.py", scene, f"--iteration {iteration}", *extra_args)


_extra_scene_args = dict(
    [(scene, ["-i images_4"]) for scene in SCENESETS[Scenesets.MipNeRF360Outdoor]]
    + [(scene, ["-i images_2"]) for scene in SCENESETS[Scenesets.MipNeRF360Indoor]]
    + [(scene, ["-i images_4"]) for scene in SCENESETS[Scenesets.NeRFLLFF]]
)


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--nopose", action="store_true")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--with_validation", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--dryrun", action="store_true")
parser.add_argument("--output_path", default=GS_RUNDIR, type=Path)
parser.add_argument("--extra_train_args", nargs="+", default=[])
parser.add_argument("--extra_render_args", nargs="+", default=[])
parser.add_argument("--extra_eval_args", nargs="+", default=[])
parser.add_argument("--extra_scene_args", nargs="+", default=[])
parser.add_argument("--scenes", nargs="+", default=[])

add_datasets_to_parser(parser)
args = parser.parse_args()

DRYRUN |= args.dryrun
QUIET |= args.quiet

args.extra_train_args = [f"--{arg}" for arg in args.extra_train_args]
args.extra_render_args = [f"--{arg}" for arg in args.extra_render_args]
args.extra_eval_args = [f"--{arg}" for arg in args.extra_eval_args]
def _add_extra_args(args):
    for arg in args.extra_scene_args:
        scene, new_extra_args = arg.split(":")
        extra_args = _extra_scene_args.get(scene, [])
        _extra_scene_args[scene] = extra_args + new_extra_args.split()
_add_extra_args(args)

configs: list[Config] = []
for ds in DATASETS:
    dataset_path = getattr(args, ds.value)
    if dataset_path is not None:
        configs += [Config(ds.name, DATASETS[ds], dataset_path)]

train_script = "train_nopose.py" if args.nopose else "train.py"
dispatcher = Dispatcher(train_script, args.output_path, args.overwrite)

if not args.skip_training:
    common_args = ["--quiet", "--eval"]
    if not args.with_validation and not args.nopose:
        common_args += ["--test_iterations -1"]
    common_args += args.extra_train_args

    for config in configs:
        for scene in config.scenes:
            if args.scenes and scene not in args.scenes:
                continue
            extra_args = _extra_scene_args.get(scene, []) + common_args
            dispatcher.run_training(config.source / scene, extra_args)

if not args.skip_rendering:
    common_args = ["--quiet", "--eval",  "--skip_train"]
    common_args += args.extra_render_args

    for config in configs:
        for scene in config.scenes:
            if args.scenes and scene not in args.scenes:
                continue
            scene_path = config.source / scene
            # dispatcher.run_rendering(scene_path, 7000, common_args)
            dispatcher.run_rendering(scene_path, -1, common_args)

if not args.skip_metrics:
    scenes_string = ""
    for config in configs:
        for scene in config.scenes:
            if args.scenes and scene not in args.scenes:
                continue
            scenes_string += f'"{args.output_path / Path(scene).name}" '

    run_cmd(f"python metrics.py -m {scenes_string} {' '.join(args.extra_eval_args)}")
