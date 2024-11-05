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
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from arguments import ModelParams
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal
from utils.sh_utils import SH2RGB

from tqdm.contrib.concurrent import thread_map
from operator import itemgetter


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    K: list[float]
    image: np.ndarray | Image.Image
    depth: np.ndarray | Image.Image | None
    image_path: Path
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud | None
    train_cameras: list[CameraInfo]
    test_cameras: list[CameraInfo]
    nerf_normalization: dict
    ply_path: Path


def getNerfppNorm(cam_infos: list[CameraInfo]):
    def get_center_and_diag(cam_centers: npt.NDArray):
        center = np.mean(cam_centers, axis=0)
        dist = np.linalg.norm(cam_centers - center, axis=1)
        diagonal = np.max(dist)
        return center, diagonal

    cam_centers = np.array([c.T for c in cam_infos])
    center, diagonal = get_center_and_diag(cam_centers)

    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}


def init_point_cloud(ply_path: Path, scale: float, bias: float, num_pts: int=100_000):
    # Generate random points
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (2 * np.random.random((num_pts, 3)) - 1) * scale + bias
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )

    storePly(ply_path, xyz, SH2RGB(shs) * 255)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T

    def _try_load(*keys):
        try:
            return np.vstack([vertices[k] for k in keys]).T
        except:
            print(f"Warning: pointcloud has not all requested keys {keys}")
            return None
    
    colors = _try_load("red", "green", "blue") / 255.0
    normals = _try_load("nx", "ny", "nz")
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: Path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(str(path))


class Reader:
    def __init__(self, train_cams, test_cams, depth_mode=None):
        self.depth_mode = depth_mode
        self._train_cams = train_cams
        self._test_cams = test_cams

    @staticmethod
    def split_cams(cams: list, test_hold: float|int=8):
        if test_hold == 0:
            train_cams, test_cams = cams, []
        elif test_hold > 1:
            llffhold = int(test_hold)
            train_cams = [c for idx, c in enumerate(cams) if idx % llffhold != 0]
            test_cams = [c for idx, c in enumerate(cams) if idx % llffhold == 0]
        else:
            test_hold = float(test_hold)
            num_val = int(test_hold * len(cams))
            print(f"Using {num_val} last images for validation")
            train_cams = cams[:-num_val]
            test_cams = cams[-num_val:]
        return train_cams, test_cams

    def read_cam(self, cam):
        raise(NotImplementedError)

    def read_depth(self, depth_dir: Path, image_name: str):
        if self.depth_mode:
            depth = Image.open(depth_dir / self.depth_mode / f"{image_name}_depth.png")
        else:
            depth = None
        return depth

    def load_train(self):
        return self._load(self._train_cams, "Loading training views")

    def load_test(self):
        return self._load(self._test_cams, "Loading testing views")

    def _load(self, cams, desc=None) -> list[CameraInfo]:
        return thread_map(self.read_cam, cams, desc=desc)


def readSceneInfo(path: Path, eval: bool, reader: Reader):
    train_cam_infos = reader.load_train()
    test_cam_infos = reader.load_test()

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if (path / "sparse/0/").exists():
        ply_path = path / "sparse/0/points3D.ply"
        if not ply_path.exists():
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(ply_path.with_suffix(".bin"))
            except:
                xyz, rgb, _ = read_points3D_text(ply_path.with_suffix(".txt"))
            storePly(ply_path, xyz, rgb)
    else:
        ply_path = path / "points3d.ply"
        if not ply_path.exists():
            ply_path = path / "points3D.ply"
        if not ply_path.exists():
            scale, bias = nerf_normalization["radius"], nerf_normalization["translate"]
            init_point_cloud(ply_path, scale, bias)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


class ColmapReader(Reader):
    def __init__(self, path: Path, depth_mode=None, image_dir=None, test_hold: float|int=8):
        try:
            cameras_extrinsic_file = path / "sparse/0/images.bin"
            cameras_intrinsic_file = path / "sparse/0/cameras.bin"
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = path / "sparse/0/images.txt"
            cameras_intrinsic_file = path / "sparse/0/cameras.txt"
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        image_dir = image_dir or "images"
        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsics
        self.images_folder = path / image_dir

        cams = [(key, Path(ex.name).stem) for key, ex in cam_extrinsics.items()]
        cams = [c for c, _ in sorted(cams, key=itemgetter(1))]

        train_cams, test_cams = self.split_cams(cams, test_hold)
        super().__init__(train_cams, test_cams, depth_mode)

    def read_cam(self, key):
        extr = self.cam_extrinsics[key]
        intr = self.cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # we want the cam pose, so invert colmap's world-to-cam transform
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = -R @ np.array(extr.tvec)

        # TODO: properly handle camera models
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            fx, cx, cy = intr.params[:3]
            fy = fx
        elif intr.model == "PINHOLE" or intr.model == "RADIAL":
            fx, fy, cx, cy = intr.params[:4]
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = self.images_folder / Path(extr.name).name
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGB")
        depth = self.read_depth(self.images_folder.parent, image_name)

        scale_x, scale_y = image.width / intr.width, image.height / intr.height
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            K=[fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y],
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        return cam_info


class BlenderReader(Reader):
    def __init__(self, path: Path, depth_mode=None, extension=".png"):
        self.path = path
        self.extension = extension
        train_frames, fovx, fovy = self._read_transformsfile(path / "transforms_train.json")
        test_frames, *_ = self._read_transformsfile(path / "transforms_test.json")
        # TODO: handle different train/test intrinsics
        self.fovx, self.fovy = fovx, fovy
        super().__init__(train_frames, test_frames, depth_mode)
    
    def _read_transformsfile(self, transformsfile):
        with open(transformsfile) as json_file:
            contents = json.load(json_file)
            # TODO: handle other camera intrinsics, if available
            fovx = contents["camera_angle_x"]
            fovy = contents.get("camera_angle_y", fovx)

            frames = list(enumerate(contents["frames"]))
        return frames, fovx, fovy

    def _with_suffix(self, filename):
        if "." in filename[-4:]:
            return filename
        else:
            return filename + self.extension

    def read_cam(self, frame):
        idx, frame = frame
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        R = c2w[:3, :3]
        T = c2w[:3, 3]

        image_path = self.path / self._with_suffix(frame["file_path"])
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGBA")
        depth = self.read_depth(self.path, image_name)

        fx = fov2focal(self.fovx, image.size[0])
        fy = fov2focal(self.fovy, image.size[1])
        # TODO: handle different focal lengths + center points
        cx = image.size[0] / 2
        cy = image.size[1] / 2

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            K=[fx, fy, cx, cy],
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=image.size[0],
            height=image.size[1],
        )


class ScannetReader(Reader):
    def __init__(self, path: Path, depth_mode=None, test_hold: float|int=8, img_skip=10):
        scene_root = path / "frames"
        nframes = len(list((scene_root / "color").glob("*.jpg")))
        K = np.loadtxt(scene_root / "intrinsic/intrinsic_color.txt")
        # intrinsics are given for the original resolution, given in <scanid>.txt
        # the scans we use had a resolution of 1296x968, so just use that
        # images on disk are 640x480 => scale intrinsics accordingly
        scale_x, scale_y = 640/1296, 480/968
        K = [scale_x * K[0, 0], scale_y * K[1, 1], scale_x * K[0, 2], scale_y * K[1, 2]]

        self.scene_root = scene_root
        self.K = K

        frames = list(range(0, nframes, img_skip))
        train_frames, test_frames = self.split_cams(frames, test_hold)
        super().__init__(train_frames, test_frames, depth_mode)

    def read_cam(self, frame_id):
        # TODO: remove invalid cameras
        # cam_infos = list(filter(lambda c: np.all(np.isfinite(c.R)) and np.all(np.isfinite(c.T)), cam_infos))
        pose = np.loadtxt(self.scene_root / f"pose/{frame_id}.txt")
        image_path = self.scene_root / f"color/{frame_id}.jpg"
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGB")
        depth = self.read_depth(self.scene_root.parent, image_name)
        width, height = image.size
        return CameraInfo(
            uid=frame_id,
            R=pose[:3, :3],
            T=pose[:3, 3],
            K=self.K.copy(),
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )


class TUMReader(Reader):
    def __init__(self, path, depth_mode, test_hold: float|int=8, max_difference=0.01):
        # from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
        # fx, fy, cx, cy
        INTRINSICS = {
            "freiburg1": [517.3, 516.5, 318.6, 255.3],
            "freiburg2": [520.9, 521.0, 325.1, 249.7],
            "freiburg3": [535.4, 539.2, 320.1, 247.6],
            "default": [525.0, 525.0, 319.5, 239.5],
        }
        K = INTRINSICS.get(path.name.split("_")[2], INTRINSICS["default"])

        def _readlines(file):
            return [
                l.strip().split(" ")
                for l in file if not l.startswith("#") and l.strip() != ""
            ]
        rgbs = _readlines(open(path / "rgb.txt"))
        poses = _readlines(open(path / "groundtruth.txt"))

        rgb_timestamps = np.array([float(l[0]) for l in rgbs])
        pose_timestamps = np.array([float(l[0]) for l in poses])

        def associate(ts_rgb, ts_pose, max_dt):
            associations = []
            for i, t in enumerate(ts_rgb):
                j = np.argmin(np.abs(ts_pose - t))
                if np.abs(ts_pose[j] - t) < max_dt:
                    associations.append((i, j))
            return np.array(associations)

        self.K = K
        self.rgbs = rgbs
        self.poses = poses
        self.timestamps = associate(rgb_timestamps, pose_timestamps, max_difference)
        self.path = path

        frames = list(range(len(self.timestamps)))
        train_frames, test_frames = self.split_cams(frames, test_hold)
        super().__init__(train_frames, test_frames, depth_mode)
    
    def read_cam(self, frameid):
        rgbid, poseid = self.timestamps[frameid]
        image_path = self.path / self.rgbs[rgbid][1]
        image_name = image_path.stem
        quat_pose = np.array([float(v) for v in self.poses[poseid][1:]])
        pose_mat = np.eye(4)
        pose_mat[:3, 3] = quat_pose[:3]
        pose_mat[:3, :3] = Rotation.from_quat(quat_pose[3:]).as_matrix()
        
        image = Image.open(image_path).convert("RGB")
        depth = self.read_depth(self.path, image_name)
        width, height = image.size
        return CameraInfo(
            uid=frameid,
            R=pose_mat[:3, :3],
            T=pose_mat[:3, 3],
            K=self.K.copy(),
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )


class ReplicaReader(Reader):
    def __init__(self, path, depth_mode=None, test_hold: float|int=8, imgskip=5):
        fx, fy, cx, cy = 600.0, 600.0, 600.0, 340.0

        self.path = path
        self.K = [fx, fy, cx, cy]
        self.poses = [
            np.array([float(v) for v in l.split(" ")]).reshape(4, 4)
            for l in open(path / "traj.txt").readlines()
        ]

        frames = list(range(0, len(self.poses), imgskip))
        train_frames, test_frames = self.split_cams(frames, test_hold)
        super().__init__(train_frames, test_frames, depth_mode)
    
    def read_cam(self, i):
        p = self.poses[i]
        image_path = self.path / f"results/frame{i:06}.jpg"
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGB")
        depth = self.read_depth(self.path, image_name)
        width, height = image.size
        return CameraInfo(
            uid=i,
            R=p[:3, :3],
            T=p[:3, 3],
            K=self.K.copy(),
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )


class ScannetPPReader(Reader):
    def __init__(self, path, depth_mode=None):
        with open(path / "dslr/nerfstudio/transforms_undistorted.json") as f:
            scene_info = json.load(f)
        # undistorted images should be converted to pinhole camera intrinsics
        self.K = [scene_info.get(k, None) for k in ("fl_x", "fl_y", "cx", "cy")]
        self.scene_info = scene_info
        self.path = path

        train_cams = list(enumerate(scene_info["frames"]))
        test_cams = list(enumerate(scene_info["test_frames"]))
        super().__init__(train_cams, test_cams, depth_mode)
    
    def read_cam(self, frame_info):
        i, frame = frame_info
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        R = c2w[:3, :3]
        T = c2w[:3, 3]
        image_path = self.path / "dslr/undistorted_images" / frame["file_path"]
        image_name = image_path.stem
        image = Image.open(image_path).convert("RGB")
        depth = self.read_depth(self.path, image_name)

        # TODO: support mask + is_bad flag
        return CameraInfo(
            uid=i,
            R=R,
            T=T,
            K=self.K,
            image=image,
            depth=depth,
            image_path=image_path,
            image_name=image_name,
            width=image.width,
            height=image.height,
        )

try:
    from co3d.dataset.data_types import (
        FrameAnnotation, load_dataclass_jgzip
    )

    class CO3DReader(Reader):
        def __init__(self, path, depth_mode=None, test_hold: float|int=8):
            annotations_file = path.parent / "frame_annotations.jgz"
            frames = [
                f for f in load_dataclass_jgzip(annotations_file, list[FrameAnnotation])
                if f.sequence_name == path.name
            ]
            frames = list(enumerate(frames))

            self.path = path
            self.frames = frames
            self.dataset_root = path.parent.parent
            train_frames, test_frames = self.split_cams(frames, test_hold)
            super().__init__(train_frames, test_frames, depth_mode)
        
        def read_cam(self, frame_info: tuple[int, FrameAnnotation]):
            i, f = frame_info
            # co3d provides data in different formats than the other datasets.
            # see https://github.com/facebookresearch/co3d/blob/main/co3d/dataset/data_types.py
            image_path = Path(self.dataset_root / f.image.path)
            image_name = image_path.stem
            image = Image.open(image_path).convert("RGB")
            depth = self.read_depth(self.path, image_name)
            height, width = f.image.size
            # co3d provides intrinsics in ndc space
            # only support isotropic ndc intrinsics for now
            assert f.viewpoint is not None
            assert f.viewpoint.intrinsics_format == "ndc_isotropic"
            scale = min(height, width) / 2
            fx, fy = f.viewpoint.focal_length
            cx, cy = f.viewpoint.principal_point
            fx *= scale
            fy *= scale
            cx = width/2 - scale * cx
            cy = height/2 - scale * cy
            R = np.array(f.viewpoint.R)
            T = np.array(f.viewpoint.T)

            # co3d provides poses in pytorch3d format:
            # X_cam = X_world @ R + T
            # we need c2w and left-multiply: X_world = R' @ X_cam + T'
            T = -R @ T
            # co3d/pytorch3d has x left, y up in camera space; we use x right, y down
            R[:, :2] *= -1

            return CameraInfo(
                uid=i,
                R=R,
                T=T,
                K=[fx, fy, cx, cy],
                image=image,
                depth=depth,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
except ImportError:
    class CO3DReader(Reader):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CO3DReader requires co3d package (https://github.com/facebookresearch/co3d) to be installed.")


def load_scene_info(args: ModelParams) -> SceneInfo:
    source_path = Path(args.source_path)
    _readers = {
        # Note that the extra args are tuples (trailing comma).
        "sparse": (ColmapReader, (args.images, args.test_hold)),
        "transforms_train.json": (BlenderReader, ()),
        "frames": (ScannetReader, (args.test_hold,)),
        "accelerometer.txt": (TUMReader, (args.test_hold,)),
        "traj.txt": (ReplicaReader, (args.test_hold,)),
        "dslr/nerfstudio/transforms_undistorted.json": (ScannetPPReader, ()),
        "../frame_annotations.jgz": (CO3DReader, (args.test_hold,)),
    }
    for subpath in _readers:
        if (source_path / subpath).exists():
            Reader, extra_args = _readers[subpath]
            reader = Reader(source_path, args.load_depth, *extra_args)
            return readSceneInfo(source_path, args.eval, reader)
    assert False, f"Could not recognize scene type in {args.source_path}"
