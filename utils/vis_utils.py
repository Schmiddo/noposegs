import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils.general_utils import align_poses


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def plot_traj(ax: plt.Axes, pred_poses, gt_poses, align=False, connect_traj=True):
    if align:
        pred_poses = align_poses(pred_poses, gt_poses)

    if isinstance(pred_poses, torch.Tensor):
        pred_poses = pred_poses.detach().cpu().numpy()
        gt_poses = gt_poses.detach().cpu().numpy()

    kwargs = {"linewidth": 3}
    if not connect_traj:
        kwargs.update(linestyle=None)
    ax.plot(*gt_poses[:, :3, 3].T, color="deepskyblue", **kwargs)
    ax.plot(*pred_poses[:, :3, 3].T, color="red", **kwargs)

    # center = gt_poses[:, :3, 3].mean(axis=(0, 1))
    # scene_size = 0.1 * np.abs(np.max((gt_poses[:, :3, 3] - center)))

    # def make_lines(poses):
    #     start, end = poses[:, :3, 3], poses[:, :3, 2]
    #     end = start + 0.1 * end
    #     dirs = np.stack([start, end], axis=1)
    #     return Line3DCollection(dirs)

    # ax.add_collection(make_lines(gt_poses))
    # ax.add_collection(make_lines(pred_poses))
