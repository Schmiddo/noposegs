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

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from cudaops import blur2d


@torch.compile(dynamic=True)
def anisotropy_loss(scalings, r):
    return (scalings.max(dim=-1).values/scalings.min(dim=-1).values - r).clamp(0).mean()


@torch.compile()
def _smoothness_loss(prediction, image):
    # Just ignore the upper-left image border
    pred_dy = prediction[..., 1:, 1:] - prediction[..., :-1, 1:]
    pred_dx = prediction[..., 1:, 1:] - prediction[..., 1:, :-1]

    img_dy = image[..., 1:, 1:] - image[..., :-1, 1:]
    img_dx = image[..., 1:, 1:] - image[..., 1:, :-1]

    neg_grad_mag_x = -img_dx.norm(dim=-3)
    neg_grad_mag_y = -img_dy.norm(dim=-3)

    m = pred_dx.abs() * neg_grad_mag_x.exp() + pred_dy.abs() * neg_grad_mag_y.exp()
    return m.mean()


def smoothness_loss(prediction, image):
    prediction = blur2d(prediction[None, ...], sigma=1.0, window_size=5)
    image = blur2d(image[None, ...], sigma=1.0, window_size=5)
    return _smoothness_loss(prediction, image)

@torch.compile()
def _masked_l1_loss(a, b, m):
    return (m * (a - b)).abs().mean()

@torch.compile()
def _l1_loss(a, b):
    return (a - b).abs().mean()

def l1_loss(network_output: torch.Tensor, gt: torch.Tensor, mask=None):
    if mask is None:
        return _l1_loss(network_output, gt)
    else:
        return _masked_l1_loss(network_output, gt, mask)


@torch.compile()
def l2_loss(network_output: torch.Tensor, gt: torch.Tensor):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size: int, sigma: float):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


_SSIM_WINDOW_CACHE = {}
@torch.no_grad()
def create_window(
    window_size: int,
    channel: int,
    device: torch.device=None,
    dtype: torch.dtype=torch.float
):
    key = (window_size, channel, device, dtype)
    if key not in _SSIM_WINDOW_CACHE:
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        )
        window = window.to(device, dtype)
        _SSIM_WINDOW_CACHE[key] = window
    return _SSIM_WINDOW_CACHE[key]


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    mask=None,
):
    if img1.ndim == 3: img1 = img1[None, ...]
    if img2.ndim == 3: img2 = img2[None, ...]

    return _ssim2(img1, img2, window_size=window_size, mask=mask)


@torch.compile()
def _ssim_map(mu1, mu2, sq1, sq2, sq12):
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (sq1 - mu1_sq)
    sigma2_sq = (sq2 - mu2_sq)
    sigma12 = (sq12 - mu1_mu2)

    C1 = 0.01**2
    C2 = 0.03**2

    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )


def _ssim2(
    img1: torch.Tensor,
    img2: torch.Tensor,
    sigma: float=1.5,
    window_size: int=11,
    mask=None,
):
    mu1 = blur2d(img1, sigma, window_size)
    mu2 = blur2d(img2, sigma, window_size)
    sq1 = blur2d(img1 * img1, sigma, window_size)
    sq2 = blur2d(img2 * img2, sigma, window_size)
    sq12 = blur2d(img1 * img2, sigma, window_size)

    ssim_map = _ssim_map(mu1, mu2, sq1, sq2, sq12)
    if mask is not None:
        ssim_map = mask * ssim_map

    return ssim_map.mean()


@torch.compile()
def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()
