import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene.cameras import Camera

from utils.loss_utils import ssim


def test_time_opt(cam: Camera, gaussians, pipeline, background, lr=1e-3, steps=100):
    cam = cam.clone()
    with torch.enable_grad():
        optimizer = torch.optim.Adam(cam.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 1e-4)
        t = tqdm(range(steps), desc="Test-time opt", leave=False)
        for i in t:
            optimizer.zero_grad()
            render_pkg = render(cam, gaussians, pipeline, background)
            img, depth = render_pkg["render"], render_pkg["depth"]
            gt_img = cam.image_with_background(background)
            loss = 0.8 * (gt_img - img).abs().mean() + 0.2 * (1 - ssim(img, gt_img))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 20 == 0:
                t.set_description(f"loss {loss.item():.5f}")
                t.refresh()
            cam.update_pose()
    return cam
