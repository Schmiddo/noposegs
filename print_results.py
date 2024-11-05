import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


from utils.data_utils import DATASETS, add_datasets_to_parser, get_datasets_from_args


METRICS = ["PSNR", "SSIM", "LPIPS", "RPE_t", "RPE_r", "ATE", "ROT", "POS"]
ABS_POSE_METRICS = ["rot@5", "pos@0.05", "rot@0.1", "pos@0.01"]


def read_dataset_results(root: Path, scenes: list[str]):
    results = {
        scene: json.load(open(root / scene / "results.json"))
        for scene in scenes
        if (root / scene / "results.json").exists()
    }
    if results and not isinstance(next(iter(next(iter(results.values())).values())), dict):
        results = {k: {"ours": v} for k, v in results.items()}
    return results

def make_table(results, metrics=None, format="plain"):
    metrics = metrics or METRICS
    formats = {
        "plain": (" ", ""),
        "latex": (" & ", "\\\\"),
    }
    sep, eol = formats[format]
    header = sep.join([f"{m:>10}" for m in metrics])
    print(f"{'Method':15}{sep}{'Scene':20}{sep}{header} {eol}")
    def print_line(method: str, scene: str, values: dict[str, float]):
        line = sep.join([f"{values[metric]:>10.3f}" for metric in metrics])
        print(f"{method:15}{sep}{scene:20}{sep}{line} {eol}")
    for scene in results:
        for method in results[scene]:
            values = results[scene][method]
            for m in ["RPE_t", "POS"]:
                if m in values:
                    values[m] *= 100
            print_line(method, scene, values)

    means = defaultdict(lambda: {m: 0 for m in metrics})
    s = 1 / len(results)
    for scene in results:
        for method in results[scene]:
            for m in means[method]:
                means[method][m] += s * results[scene][method][m]
    for method in means:
        print_line(method, "mean", means[method])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", default=Path("."), type=Path)
    parser.add_argument("--format", default="plain", choices=["plain", "latex"])
    parser.add_argument("--scale_rpet", type=float, default=1)
    parser.add_argument("--metrics", default=[], nargs="+", choices=METRICS + ABS_POSE_METRICS)
    add_datasets_to_parser(parser)
    args = parser.parse_args()

    datasets = get_datasets_from_args(args)

    for dataset in datasets:
        results = read_dataset_results(args.root, [Path(s).name for s in DATASETS[dataset]])
        if not results:
            print("Found no results!")
            exit(0)
        make_table(results, metrics=args.metrics, format=args.format)
        print()
