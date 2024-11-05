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

import shutil
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams
from scene.cameras import Camera
from scene.dataset_readers import load_scene_info
from utils.camera_utils import cameraList_from_camInfos, save_caminfos
from utils.general_utils import (
    get_lr_scheduler,
    pose_error,
    procrustes,
    read_poses,
    se3_from_mat4,
    save_cam_poses,
)
from utils.system_utils import searchForMaxIteration

from .pose import random_pose


class Scene:
    def __init__(
        self, args: ModelParams, load_iteration: int|None=None
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = Path(args.model_path)
        self.loaded_iter = None

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    str(self.model_path / "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        scene_info = load_scene_info(args)

        if not self.loaded_iter:
            shutil.copy(scene_info.ply_path, str(self.model_path / "input.ply"))
            caminfo_path = self.model_path / "cameras.json"
            save_caminfos(
                caminfo_path, scene_info.train_cameras, scene_info.test_cameras
            )

        self.point_cloud = scene_info.point_cloud
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")

        def _filter_views(cams: list[Camera]):
            return [c for c in cams if c.image_name in args.view_list]

        cams = cameraList_from_camInfos(scene_info.train_cameras, args)

        if hasattr(args, "view_list") and args.view_list:
            cams = _filter_views(cams)
            print(f"Filtered view list contains {len(cams)} views")
        if args.init_cam_identity:
            for c in cams:
                c._pose.data.copy_(Camera.Group.id_elem.to(c._pose.device))
        if hasattr(args, "cam_noise") and args.cam_noise > 0:
            print(f"Adding noise with std {args.cam_noise}")
            for c in cams:
                c._pose = c._pose * random_pose(args.cam_noise).to(c._pose.device)
        self.train_cameras = cams

        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        if self.loaded_iter is not None:
            cam_path = self.model_path / f"train_cameras_{self.loaded_iter}.pth"
            if cam_path.exists():
                print("Loading trained poses")
                gt_poses, poses = read_poses(cam_path)
                assert len(poses) == len(self.train_cameras), f"{len(poses)} != {len(self.train_cameras)}"
                for c, p in zip(self.train_cameras, poses):
                    c._pose.data = se3_from_mat4(p).cuda()


    def load(self, checkpoint):
        gt_poses, poses = read_poses(checkpoint)
        assert len(poses) == len(self.train_cameras), "Num cameras does not match"
        for c, p in zip(self.train_cameras, poses):
            c._pose.data = se3_from_mat4(p).cuda()

    def save(self, iteration):
        save_cam_poses(
            self.train_cameras, self.model_path / f"train_cameras_{iteration}.pth"
        )
        if self.test_cameras:
            save_cam_poses(
                self.test_cameras, self.model_path / f"test_cameras_{iteration}.pth"
            )


    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    @torch.no_grad()
    def getAlignedTestCameras(self):
        # we just mak all poses relative to the first and correct for scale
        gt_poses = self.gt_poses()
        pred_poses = self.pred_poses()
        T = pred_poses[0] @ torch.linalg.inv(gt_poses[0])
        gt_poses = T @ gt_poses
        s, _, _ = procrustes(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
        T_se3 = self.train_cameras[0].pose * self.train_cameras[0].gt_pose.inv()
        # align_sim3 returns a copy of the camera (with detached parameters)
        test_cams = [c.clone() for c in self.test_cameras]
        for c in test_cams:
            # move to be relative to first estimated pose
            c._pose.data((T_se3 * c.pose).data)
            # adapt scaling
            c._pose.data[:3] *= s
        return test_cams

    def gt_poses(self):
        return torch.stack([c.gt_pose.matrix() for c in self.train_cameras])

    def pred_poses(self):
        return torch.stack([c.pose.matrix() for c in self.train_cameras])

    def set_camera_scale(self, scale):
        for c in self.train_cameras:
            c.image_scale = scale
        for c in self.test_cameras:
            c.image_scale = scale

    def training_setup(self, training_args: OptimizationParams):
        if training_args.cam_lr_init > 0:
            params = [
                {
                    "params": c.parameters(),
                    "lr": training_args.cam_lr_init,
                    "name": f"cam{i}",
                }
                for i, c in enumerate(self.train_cameras)
            ]
            self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
            self.scheduler = get_lr_scheduler(
                training_args.cam_lr_scheduler,
                lr_init=training_args.cam_lr_init,
                lr_final=training_args.cam_lr_final,
                lr_delay_steps=training_args.cam_lr_delay_steps,
                lr_delay_mult=training_args.cam_lr_delay_mult,
                max_steps=training_args.cam_lr_max_steps,
            )
        else:
            self.optimizer = None
            self.scheduler = None

    def update_learning_rate(self, iteration: int):
        if self.optimizer is None:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.scheduler(iteration)
