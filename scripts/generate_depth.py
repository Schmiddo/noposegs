import os
import sys
import argparse
import torch
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm

from arguments import ModelParams
from scene.dataset_readers import load_scene_info

import sys
sys.path.append("submodules/dpt")
from submodules.dpt.midas.model_loader import load_model


DPT_CHECKPOINT = Path(os.getenv("MODELDIR", "models")) / "dpt_hybrid-midas-501f0c75.pt"


class Processor(torch.nn.Module):
    def __init__(self, predictor, transforms, scale=1.0, shift=0.0):
        super().__init__()
        self._transforms = transforms
        self._predictor = predictor
        self._scale = scale
        self._shift = shift

    def forward(self, image):
        image = self._transforms({"image": np.array(image)})["image"]
        image = torch.from_numpy(image).cuda().unsqueeze(0)
        output = self._predictor(image).float()
        output = (self._scale * output + self._shift)
        return output


def get_dpt_model(checkpoint, scale=1.0, shift=0.0):
    model, transforms, h, w = load_model("cuda", checkpoint, "dpt_hybrid_384")
    estimator = Processor(model, transforms, scale, shift)
    return estimator

def get_depth_anything_model(scale=1.0, shift=0.0):
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cuda")
    return lambda img: scale * pipe(img)["predicted_depth"] + shift


def generate_depth_single_scene(model_name, min_disp, model, opts: ModelParams):
    depth_path = Path(opts.source_path) / model_name
    depth_path.mkdir(exist_ok=True)
    scene_info = load_scene_info(opts)
    for c in tqdm(scene_info.train_cameras):
        image = c.image.convert("RGB")
        inverse_depth = model(image).squeeze()
        torch.save(inverse_depth.cpu(), depth_path / (c.image_name + "_depth.pth"))
        depth = 1 / inverse_depth.clip(min_disp)
        depth = depth * min_disp
        depth_image = depth.mul(65535).cpu().numpy().astype(np.uint16)
        iio.imwrite(depth_path / (c.image_name + "_depth.png"), depth_image)


@torch.no_grad()
def main(model_name, model, min_disp: float, scenes: list[str], opts: ModelParams):
    for scene in scenes:
        opts.source_path = str(scene)
        generate_depth_single_scene(model_name, min_disp, model, opts)


if __name__ == "__main__":
    from utils.data_utils import (
        add_datasets_to_parser, get_datasets_from_args, DATASETS, DATA_DIR, DATASET_ARGS
    )

    parser = argparse.ArgumentParser()
    opts = ModelParams.add(parser)
    add_datasets_to_parser(parser)
    parser.add_argument("--model", default="dpt", choices=["dpt", "depth-anything"])
    parser.add_argument("--checkpoint", default=DPT_CHECKPOINT, type=Path)
    parser.add_argument("--scale", type=float, default=0.000305)
    parser.add_argument("--shift", type=float, default=0.1378)
    parser.add_argument("--scenes", nargs="+", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    opts.eval = False

    if args.shift <= 0:
        raise(f"shift must be strictly positive.")

    if args.model == "dpt":
        model = get_dpt_model(args.checkpoint, args.scale, args.shift).cuda()
    else:
        model = get_depth_anything_model(args.scale, args.shift)    

    datasets = get_datasets_from_args(args)
    if args.all:
        datasets = list(DATASETS.keys())
    print("DAtasets:", datasets)
    scenes = [DATA_DIR / DATASET_ARGS[d].path / scene for d in datasets for scene in DATASETS[d]]
    
    if args.scenes:
        scenes = [Path(opts.source_path) / scene for scene in args.scenes]
    elif opts.source_path:
        scenes = [opts.source_path]
    if not scenes:
        print("No scenes specified")
        exit(1)

    print(f"Computing depth for {len(scenes)} scenes")
    main(args.model, model, args.shift, scenes, opts.extract(args))
