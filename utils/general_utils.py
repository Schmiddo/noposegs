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

import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image: PILImage, resolution=None, value_range=255):
    if resolution is not None:
        pil_image = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image).astype(np.float32)) / value_range
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


class Scheduler:
    def __init__(self, optimizer, schedule, predicate=None):
        self._optimizer = optimizer
        self._schedule = schedule
        self._predicate = predicate or (lambda _: True)

    def step(self, iteration):
        for param_group in self._optimizer.param_groups:
            if self._predicate(param_group):
                param_group["lr"] = self._schedule(iteration)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_cosine_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        lr_cur = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t))
        return delay_rate * lr_cur

    return helper


def get_lr_scheduler(
    type, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    if type == "exp":
        return get_expon_lr_func(
            lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps
        )
    elif type == "cosine":
        return get_cosine_lr_func(
            lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps
        )
    elif type == "none":
        return lambda _: lr_init


def strip_lowerdiag(L: torch.Tensor):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # kidnapped from pytorch3d
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def get_quat(matrix: torch.Tensor) -> torch.Tensor:
    # kidnapped from pytorch3d
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def build_rotation(r: torch.Tensor):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent, seed: int=0):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

        def isatty(self):
            return old_f.isatty()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))


def normq(q: torch.Tensor):
    return q.mul(q.square().sum(-1).rsqrt())


def se3_from_mat4(mat4):
    q = get_quat(mat4[:3, :3])
    lietorch_q = torch.cat([q[1:], q[:1]], dim=0)
    t = mat4[:3, 3]
    return torch.cat([t, normq(lietorch_q)], dim=0)


@torch.no_grad()
def save_cam_poses(cams: list, path):
    poses = {
        "gt": torch.stack([c.gt_pose.matrix() for c in cams]),
        "pose": torch.stack([c.pose.matrix() for c in cams]),
    }
    torch.save(poses, path)


def read_poses(pose_path, map_location=None):
    poses = torch.load(pose_path, map_location=map_location)
    return poses["gt"], poses["pose"]


def pose_matrix_inv(p: torch.Tensor):
    Rinv = p[..., :3, :3].transpose(-1, -2)
    t = p[..., :3, 3:4]
    inv = torch.eye(4, device=p.device).repeat(p.shape[:-2] + (1, 1))
    inv[..., :3, :3] = Rinv
    inv[..., :3, 3:4] = -Rinv @ t
    return inv


def relative_rotation_error(rel_pose):
    trace = rel_pose[..., 0, 0] + rel_pose[..., 1, 1] + rel_pose[..., 2, 2]
    rot_error = ((trace - 1) / 2).clamp(-1, 1).acos().mul(180 / np.pi)
    return rot_error


def pose_error(
    a: torch.Tensor,
    b: torch.Tensor,
    align_poses: bool = False,
):
    if align_poses:
        b = b.clone()
        s, R, t = procrustes(a[..., :3, 3], b[..., :3, 3])
        # sim3 = procrustes(a[..., :3, 3], b[..., :3, 3])
        # effectively sim3 @ b
        b[..., :3, :3] = R @ b[..., :3, :3]
        b[..., :3, 3] = s * b[..., :3, 3] @ R.T + t
    pose_diff = pose_matrix_inv(a) @ b
    # pos_error: torch.Tensor = pose_diff[..., :3, 3].norm(dim=-1)
    pos_error: torch.Tensor = (a[..., :3, 3] - b[..., :3, 3]).norm(dim=-1)
    rot_error = relative_rotation_error(pose_diff)
    return rot_error, pos_error


def ate(gt, pred):
    squared_errors = (gt[:, :3, 3] - pred[:, :3 ,3]).square().sum(dim=-1)
    return squared_errors.mean().sqrt()


def procrustes(pts1: torch.Tensor, pts2: torch.Tensor):
    assert pts1.shape == pts2.shape, f"{pts1.shape} != {pts2.shape}"
    assert pts1.shape[-1] == 3 and len(pts1.shape) == 2, f"{pts1.shape}"
    # estimate a sim3 transformation to align two point clouds
    # find M = argmin ||P1 - M @ P2||
    t1 = pts1.mean(dim=0)
    t2 = pts2.mean(dim=0)
    pts1 = pts1 - t1[None, :]
    pts2 = pts2 - t2[None, :]
    s1 = pts1.square().sum(dim=-1).mean().sqrt()
    s2 = pts2.square().sum(dim=-1).mean().sqrt()
    pts1 = pts1 / s1
    pts2 = pts2 / s2
    try:
        U, _, V = (pts1.T @ pts2).double().svd()
        U: torch.Tensor = U
        V: torch.Tensor = V
    except:
        print("Procustes failed: SVD did not converge!")
        s = s1 / s2
        return 1, torch.eye(3, device=pts1.device), torch.zeros_like(t1)
    # build rotation matrix
    R = (U @ V.T).float()
    if R.det() < 0:
        R[:, 2] *= -1
    s = s1 / s2
    t = t1 - s * t2 @ R.T

    # use as mat4: [sR, t] @ pts2
    # or as s * R @ pts2 + t
    return s, R, t


def align_poses(pred_poses, gt_poses) -> torch.Tensor:
    # aligns predictions to ground truth with procrustes
    pred_poses = pred_poses.clone()
    s, R, t = procrustes(gt_poses[:, :3, 3], pred_poses[:, :3, 3])
    pred_poses[:, :3, :3] = R @ pred_poses[:, :3, :3]
    pred_poses[:, :3, 3] = s * pred_poses[:, :3, 3] @ R.transpose(-1, -2) + t
    return pred_poses


def compute_traj_metrics(pred_poses, gt_poses):
    aligned_pred = align_poses(pred_poses, gt_poses)
    traj_error = ate(gt_poses, aligned_pred)

    rel_pred = torch.linalg.inv(aligned_pred[1:]) @ aligned_pred[:-1]
    rel_gt = torch.linalg.inv(gt_poses[1:]) @ gt_poses[:-1]
    rpe_r, rpe_t = pose_error(rel_gt, rel_pred)
    return rpe_r, rpe_t, traj_error


def rot_y(phi, device=None):
    return torch.tensor([
        [np.cos(phi), 0, -np.sin(phi), 0],
        [0, 1, 0, 0],
        [np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def rot_x(phi, device=None):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def rot_z(phi, device=None):
    return torch.tensor([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def trans(t, device=None):
    T = torch.eye(4, device=device)
    T[:3, -1] = t
    return T


def euler_rotation(x, y, z, device=None):
    return rot_z(z/180*np.pi, device) @ rot_y(y/180*np.pi, device) @ rot_x(x/180*np.pi, device)


def unproject_points(depth, proj, mask=None):
    # unproject points using the given depth map and projection matrix
    H, W = depth.shape[-2:]

    xy = torch.stack([
        *torch.meshgrid(
            torch.linspace(-1, 1, W, device=depth.device),
            torch.linspace(-1, 1, H, device=depth.device),
            indexing="xy",
        ),
        torch.ones(depth.shape[-2:], device=depth.device),
    ])

    if mask is not None:
        xy = xy[:, mask > 0]
        depth = depth[:, mask > 0]
    proj2 = proj[:3, :3]
    proj2[2, 2] = 1
    invproj = torch.linalg.inv(proj2)
    # invproj = torch.linalg.inv(proj)[:3, :3]
    # invproj[2, 2] = 1
    xyz = invproj @ (depth * xy).flatten(1)
    return xyz


@torch.no_grad()
def spherify_poses(poses):
    # Adapted from Nope-Nerf
    device = poses.device
    poses = poses.cpu().numpy()
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    def normalize(v):
        return v / np.linalg.norm(v)

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    return torch.as_tensor(poses_reset).float().to(device)

def align_forward(pose_a: torch.Tensor, pose_b: torch.Tensor):
    # it's like procrustes, but we care more about the orientation than the position
    pts_a = pose_a[:, :3, 3]
    pts_b = pose_b[:, :3, 3]

    t1 = pts_a.mean(dim=0)
    t2 = pts_b.mean(dim=0)
    pts_a = pts_a - t1[None, :]
    pts_b = pts_b - t2[None, :]
    s1 = pts_a.square().sum(dim=-1).mean().sqrt()
    s2 = pts_b.square().sum(dim=-1).mean().sqrt()
    pts_a = pts_a / s1
    pts_b = pts_b / s2

    z_a = F.normalize(pose_a[:, :3, 2].mean(dim=0), dim=-1)
    z_b = F.normalize(pose_b[:, :3, 2].mean(dim=0), dim=-1)
    v = torch.linalg.cross(z_b, z_a, dim=-1)
    c = torch.dot(z_b, z_a)
    def _skew(v):
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], device=v.device)
    skew_v = _skew(v)
    R = torch.eye(3, device=v.device) + skew_v + skew_v @ skew_v * (1 / (1 + c))
    s = s1 / s2
    t = t1 - s * t2 @ R.T

    # use as mat4: [sR, t] @ pts2
    # or as s * R @ pts2 + t
    return s, R, t
