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

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.functional as TF
from lietorch import SE3
from torch import nn
from einops import reduce

from utils.graphics_utils import getCVProjectionMatrix
from utils.general_utils import se3_from_mat4

from .pose import get_pose_data


def blend(image: torch.Tensor, mask: torch.Tensor, bg: torch.Tensor):
    blended = image * mask + bg[:, None, None] * (1 - mask)
    return blended


class Camera(nn.Module):
    Group = SE3

    def __init__(
        self,
        colmap_id: str,
        R: npt.NDArray,
        T: npt.NDArray,
        K: list[float],
        image: torch.Tensor,
        gt_alpha_mask: torch.Tensor | None,
        depth: torch.Tensor | None,
        image_name: str,
        uid: int,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id

        T: torch.Tensor = torch.as_tensor(T).float().cuda()
        R: torch.Tensor = torch.as_tensor(R).float().cuda()
        self.R, self.T = R, T
        self.K = K
        self.gt_pose = self.Group(get_pose_data(R, T))
        self._pose = self.Group(get_pose_data(R, T))
        self._pose_delta = nn.Parameter(torch.zeros(self.Group.manifold_dim).cuda())

        self.image_name = image_name

        self.original_image = image.clamp(0.0, 1.0).cuda().contiguous()
        self.image_scale = 1
        self.background = torch.ones(3, device="cuda")

        if gt_alpha_mask is not None:
            self.original_alpha_mask = gt_alpha_mask.cuda().contiguous()
        else:
            self.original_alpha_mask = None

        if depth is not None:
            self.original_depth = depth.cuda().contiguous()
            self.depth_bg = 1e6 * torch.ones(1, device="cuda")
        else:
            self.original_depth = None

        self.zfar = 100.0
        self.znear = 0.01

        self._projection_matrix = getCVProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fx=self.K[0], fy=self.K[1],
            cx=self.K[2], cy=self.K[3],
            width=image.shape[2], height=image.shape[1],
        ).cuda()
    
    @property
    def state(self):
        # todo: merge with state_dict functionality?
        return {
            "image": self.original_image,
            "mask": self.original_alpha_mask,
            "R": self.R,
            "T": self.T,
            "gt_pose": self.gt.data,
            "pose": self.pose.data,
            "colmap_id": self.colmap_id,
            "uid": self.uid,
            "fovx": self.FoVx,
            "fovy": self.FoVy,
            "image_name": self.image_name,
            "image_scale": self.image_scale,
        }

    @classmethod
    def from_state(cls, state):
        cam = Camera(
            state["colmap_id"],
            state["R"], state["T"], state["fovx"], state["fovy"],
            state["image"], state["mask"], state["image_name"],
            state["uid"],
        )
        cam.gt_pose.data.copy_(state["gt_pose"].data.cuda())
        cam.pose.data.copy_(state["pose"].data.cuda())
        return cam

    @property
    def FoVx(self):
        return 2 * np.arctan(self.image_width / (2 * self.K[0]))

    @property
    def FoVy(self):
        return 2 * np.arctan(self.image_height / (2 * self.K[1]))

    def _rescale(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_scale > 50:
            # Not sure if this is a good way to do it
            return reduce(
                image,
                "d (h hk) (w wk) -> d h w",
                "mean",
                hk=self.image_scale,
                wk=self.image_scale,
            )
        elif self.image_scale > 1:
            size = (self.image_height, self.image_width)
            return TF.resize(image, size, antialias=True)
        else:
            return image

    @property
    def depth(self):
        if self.original_depth is None:
            return None
        size = self.image_height, self.image_width
        if self.original_alpha_mask is not None:
            return blend(
                TF.resize(self.original_depth, size, antialias=True),
                self._rescale(self.original_alpha_mask),
                self.depth_bg,
            )
        else:
            return TF.resize(self.original_depth, size, antialias=True)

    @property
    def alpha_mask(self):
        if self.original_alpha_mask is None:
            return None
        else:
            return self._rescale(self.original_alpha_mask)

    @property
    def image_height(self):
        return int(self.original_image.shape[1] / self.image_scale)

    @property
    def image_width(self):
        return int(self.original_image.shape[2] / self.image_scale)
    
    @property
    def resolution(self):
        return (self.image_height, self.image_width)

    def _image_with_background(self, bg: torch.Tensor | None):
        if bg is None or self.original_alpha_mask is None:
            return self._rescale(self.original_image)
        else:
            return blend(
                self._rescale(self.original_image),
                self._rescale(self.original_alpha_mask),
                bg,
            )

    def image_with_background(self, bg: torch.Tensor | None):
        return self._image_with_background(bg)

    @property
    def image(self):
        return self._image_with_background(self.background)

    @property
    def pose(self) -> SE3:
        return self._pose.retr(self._pose_delta)
    
    @torch.no_grad()
    def update_pose(self):
        self._pose = self._pose.retr(self._pose_delta)
        self._pose_delta.data.zero_()

    @property
    def cam2world(self):
        return self.pose.matrix()

    @property
    def camera_center(self):
        return self.cam2world[:3, 3]

    @property
    def world2cam(self) -> torch.Tensor:
        return self.pose.inv().matrix()

    @property
    @torch.cuda.nvtx.range("projection matrix")
    def projection_matrix(self):
        return self._projection_matrix

    @property
    def full_proj_transform(self):
        return (self._projection_matrix @ self.world2cam)

    @torch.no_grad()
    def clone(self):
        # Does not preserve gradients.
        cam = Camera(
            self.colmap_id,
            self.R.detach().clone(),
            self.T.detach().clone(),
            self.K,
            self.original_image,
            self.original_alpha_mask,
            self.depth,
            self.image_name,
            self.uid,
        )
        cam.gt_pose.data.copy_(self.gt_pose.data)
        cam._pose.data.copy_(self.pose.data)
        return cam


class NoisyCamera(Camera):
    ...


class BlurryCamera(Camera):
    ...


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        projection_matrix,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.world2cam = world_view_transform
        self.projection_matrix = projection_matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.cam2world = view_inv
        self.camera_center = view_inv[:3, 3]
        self.pose = SE3(se3_from_mat4(view_inv))
