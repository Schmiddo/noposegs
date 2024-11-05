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

import numpy as np

from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

from tqdm.contrib.concurrent import thread_map

WARNED = False


def loadCam(args, id: int, cam_info: CameraInfo) -> Camera:
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (args.resolution)), round(
            orig_h / (args.resolution)
        )
        scale = args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_K = [k / scale for k in cam_info.K]

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    if cam_info.depth is not None:
        if not np.all(np.isfinite(cam_info.depth)):
            print(f"nans/infs in image {cam_info.image_name}")
        depth = PILtoTorch(cam_info.depth, value_range=65536)
        if not depth.isfinite().all():
            print(f"nans/infs in torch tensor of image {cam_info.image_name}")
    else:
        depth = None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        K=resized_K,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        depth=depth,
        image_name=cam_info.image_name,
        uid=id,
    )


def cameraList_from_camInfos(cam_infos: list[CameraInfo], args) -> list[Camera]:
    def _loadCam(id_caminfo):
        id, cam_info = id_caminfo
        return loadCam(args, id, cam_info)
    return thread_map(_loadCam, enumerate(cam_infos), disable=True)


def camera_to_JSON(id, camera: Camera):
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": camera.T.tolist(),
        "rotation": camera.R.tolist(),
        "fy": camera.K[1],
        "fx": camera.K[0],
    }
    return camera_entry


def save_caminfos(path, train_cameras, test_cameras):
    json_cams = []
    camlist = []
    if test_cameras:
        camlist.extend(test_cameras)
    if train_cameras:
        camlist.extend(train_cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(path, "w") as file:
        json.dump(json_cams, file)
