import os
from pathlib import Path

from dataclasses import dataclass
from enum import Enum
from typing import Optional


DATA_DIR = Path(os.getenv("DATA_DIR", "data/"))

# skip "flowers", "treehill"
_m360_outdoor = ["bicycle", "garden", "stump"]
_m360_indoor = ["room", "counter", "kitchen", "bonsai"]
_tandt = ["train", "truck"]
_tandt_full = ["barn", "caterpillar", "church", "courthouse", "ignatius", "meetingroom", "train", "truck"]
_tandt_nopenerf = ["Ballroom", "Barn", "Church", "Family", "Francis", "Horse", "Ignatius", "Museum"]
_db = ["drjohnson", "playroom"]
_blender = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
_llff = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
_replica = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
_scannet = [f"scene{id:04}_00" for id in [0, 59, 79, 106, 181, 207, 301, 418, 431]]
_tum = [f"rgbd_dataset_freiburg{scene}" for scene in ["1_desk2", "1_desk", "1_room", "2_xyz", "3_long_office_household"]]
_scannetpp = ["8b5caf3398", "b20a261fdf"]
# apple, bench, hydrant, skateboard, teddybear
_co3d = ["apple/110_13051_23361", "bench/415_57112_110099", "hydrant/106_12648_23157", "skateboard/245_26182_52130", "teddybear/34_1403_4393"]


class Datasets(Enum):
    MipNeRF360 = "mipnerf360"
    TanksAndTemples = "tanksandtemples"
    TanksAndTemplesFull = "tanksandtemplesfull"
    TanksAndTemplesNN = "tanksandtemplesnopenerf"
    DeepBlending = "deepblending"
    NeRFSynthetic = "nerf_synthetic"
    NeRFLLFF = "nerf_llff"
    Replica = "replica"
    ScanNet = "scannet"
    TUM_RGBD = "tum"
    ScanNetPP = "scannetpp"
    CO3D = "co3d"


class Scenesets(Enum):
    MipNeRF360Outdoor = Datasets.MipNeRF360.value + "_outdoor"
    MipNeRF360Indoor = Datasets.MipNeRF360.value + "_indoor"
    MipNeRF360 = Datasets.MipNeRF360.value
    TanksAndTemples = Datasets.TanksAndTemples.value
    TanksAndTemplesFull = Datasets.TanksAndTemples.value + "_full"
    TanksAndTemplesNN = Datasets.TanksAndTemples.value + "_nopenerf"
    DeepBlending = Datasets.DeepBlending.value
    NeRFSynthetic = Datasets.NeRFSynthetic.value
    NeRFLLFF = Datasets.NeRFLLFF.value
    Replica = Datasets.Replica.value
    ScanNet = Datasets.ScanNet.value
    TUM_RGBD = Datasets.TUM_RGBD.value
    ScanNetPP = Datasets.ScanNetPP.value
    CO3D = Datasets.CO3D.value


SCENESETS = {
    Scenesets.MipNeRF360Outdoor: _m360_outdoor,
    Scenesets.MipNeRF360Indoor: _m360_indoor,
    Scenesets.MipNeRF360: _m360_outdoor + _m360_indoor,
    Scenesets.TanksAndTemples: _tandt,
    Scenesets.TanksAndTemplesFull: _tandt_full,
    Scenesets.TanksAndTemplesNN: _tandt_nopenerf,
    Scenesets.DeepBlending: _db,
    Scenesets.NeRFSynthetic: _blender,
    Scenesets.NeRFLLFF: _llff,
    Scenesets.Replica: _replica,
    Scenesets.ScanNet: _scannet,
    Scenesets.TUM_RGBD: _tum,
    Scenesets.ScanNetPP: _scannetpp,
    Scenesets.CO3D: _co3d,
}


DATASETS = {
    Datasets.MipNeRF360: _m360_outdoor + _m360_indoor,
    Datasets.TanksAndTemples: _tandt,
    Datasets.TanksAndTemplesFull: _tandt_full,
    Datasets.TanksAndTemplesNN: _tandt_nopenerf,
    Datasets.DeepBlending: _db,
    Datasets.NeRFSynthetic: _blender,
    Datasets.NeRFLLFF: _llff,
    Datasets.Replica: _replica,
    Datasets.ScanNet: _scannet,
    Datasets.TUM_RGBD: _tum,
    Datasets.ScanNetPP: _scannetpp,
    Datasets.CO3D: _co3d,
}


@dataclass
class Dataset:
    name: str
    shorthand: str
    path: Path


DATASET_ARGS = {
    Datasets.MipNeRF360: Dataset(Datasets.MipNeRF360.value, "m360", "360_v2"),
    Datasets.TanksAndTemples: Dataset(Datasets.TanksAndTemples.value, "tat", "tandt"),
    Datasets.TanksAndTemplesFull: Dataset(Datasets.TanksAndTemplesFull.value, "tatf", "tandt"),
    Datasets.TanksAndTemplesNN: Dataset(Datasets.TanksAndTemplesNN.value, "tatnn", "Tanks"),
    Datasets.DeepBlending: Dataset(Datasets.DeepBlending.value, "db", "db"),
    Datasets.NeRFSynthetic: Dataset(Datasets.NeRFSynthetic.value, "ns", "nerf_synthetic"),
    Datasets.NeRFLLFF: Dataset(Datasets.NeRFLLFF.value, "llff", "nerf_llff_data"),
    Datasets.Replica: Dataset(Datasets.Replica.value, "rep", "Replica"),
    Datasets.ScanNet: Dataset(Datasets.ScanNet.value, "sn", "scannet"),
    Datasets.TUM_RGBD: Dataset(Datasets.TUM_RGBD.value, "tum", "TUM_RGBD"),
    Datasets.ScanNetPP: Dataset(Datasets.ScanNetPP.value, "snpp", "scannetpp/data"),
    Datasets.CO3D: Dataset(Datasets.CO3D.value, "co3d", "co3d"),
}

def add_datasets_to_parser(parser):
    for ds in DATASET_ARGS.values():
        parser.add_argument(
            f"--{ds.name}",
            f"-{ds.shorthand}",
            const=DATA_DIR / ds.path,
            default=None,
            nargs="?",
            type=Path,
        )

def get_datasets_from_args(args) -> list[Datasets]:
    return [ds for ds in DATASETS if getattr(args, ds.value) is not None]
