from typing import Any
import torch
import imageio.v3 as iio

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gaussian_renderer import render
from scene.cameras import Camera

class _DummyPipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


class AsyncImageWriter:
    _instance = None

    @classmethod
    def get(cls, num_threads=1):
        if cls._instance is None:
            cls._instance = super(AsyncImageWriter, cls).__new__(cls)
            cls._instance.executor = ThreadPoolExecutor(max_workers=num_threads)
        return cls._instance
    
    def write(self, name, img):
        self.executor.submit(self._write, name, img)
    
    @staticmethod
    def _write(path, img):
        iio.imwrite(path, img.clamp(0, 1).cpu().permute(1, 2, 0).mul(255).byte().numpy())


class CameraLogger:
    def __init__(self, root: Path, cam: Camera):
        self.cam = cam
        self.background = cam.background
        self.root = root / f"cam{self.cam.uid}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = AsyncImageWriter.get()
    
    @torch.no_grad()
    def __call__(self, gaussians, iteration):
        img = render(self.cam, gaussians, _DummyPipe(), self.background)["render"]
        self.writer.write(self.root / f"{iteration:05}.jpg", img)

class MultiCameraLogger:
    def __init__(self, root: str | Path, cams: list[Camera]) -> None:
        self.root = root
        self.loggers = [CameraLogger(Path(root), cam) for cam in cams]
    
    def __call__(self, gaussians, iteration):
        for logger in self.loggers:
            logger(gaussians, iteration)
