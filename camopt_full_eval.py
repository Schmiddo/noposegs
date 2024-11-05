import time
import os
from pathlib import Path

from dataclasses import dataclass

from utils.data_utils import Datasets, DATASETS, DATASET_ARGS, add_datasets_to_parser

@dataclass
class EvalConfig:
    datasets: list[Datasets]
    data_root: Path
    rundir_root: Path
    points_root: Path
    extra_args: list[str]
    overwrite: bool


def run_scene(dataset: str, scene: str, eval_config: EvalConfig):
    model_path = eval_config.points_root / scene / "point_cloud/iteration_30000/point_cloud.ply"
    rundir = eval_config.rundir_root / scene
    cmd = f"python cam_opt.py --eval"
    cmd = [
        cmd,
        f"-s {eval_config.data_root / dataset / scene}",
        f"-m {rundir}",
        f"--checkpoint={model_path}",
        *[f"--{c}" for c in eval_config.extra_args],
    ]
    cmd = " ".join(cmd)
    if (rundir / "results.json").exists():
        print(f"{dataset}/{scene} exists; skipping")
        return
    else:
        if os.system(cmd):
            print(f"{dataset}/{scene} failed!")
            time.sleep(2)


def print_results(eval_config: EvalConfig):
    dataset_args = " ".join([f"--{ds.value}" for ds in eval_config.datasets])
    print_results_cmd = (
        f"python print_results.py"
        f" --root {config.rundir_root}"
        f" {dataset_args}"
        f" --metrics ROT POS rot@5 pos@0.05 rot@0.1 pos@0.01"
    )
    os.system(print_results_cmd)

def main(config: EvalConfig):
    for dataset in config.datasets:
        print(f"Running camera optimization for all scenes in {dataset.name}")
        for scene in DATASETS[dataset]:
            run_scene(DATASET_ARGS[dataset].path, scene, config)
    print_results(config)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_datasets_to_parser(parser)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--rundir_root", type=Path)
    parser.add_argument("--model_root", type=Path)
    parser.add_argument("--extra_args", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.datasets = [d for d in Datasets if getattr(args, d.value, None) is not None]

    config = EvalConfig(
        args.datasets,
        args.data_root,
        args.rundir_root,
        args.model_root,
        args.extra_args,
        args.overwrite,
    )
    main(config)
