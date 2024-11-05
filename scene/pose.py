import torch
from lietorch import SE3

from utils.general_utils import get_quat


def _normalize(q: torch.Tensor):
    return q * q.square().sum(-1).rsqrt()


def random_pose(std: float):
    # important: use named parameter sigma
    return SE3.Random((), sigma=std)


def get_pose_data(R: torch.Tensor, t: torch.Tensor):
    q = get_quat(R)
    # lietorch has real value last
    lietorch_q = torch.cat([q[..., 1:], q[..., :1]], dim=-1)
    lietorch_q = _normalize(lietorch_q)
    # lietorch_q[-1] = 1
    return torch.cat([t, lietorch_q], dim=-1)
