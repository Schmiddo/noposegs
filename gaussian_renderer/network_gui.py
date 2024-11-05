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

import torch
import traceback
import json
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix

import asyncio
import time
from threading import Thread

from . import render
from arguments import PipelineParams
from utils.graphics_utils import focal2fov

_pipe = PipelineParams


def init_network_gui(source_path, host="localhost", port=8012):
    server_thread = NetworkGUI(source_path, host, port)
    server_thread.daemon = True
    server_thread.start()
    return server_thread


class NetworkGUI(Thread):
    def __init__(self, source_path, host="localhost", port=8012):
        super().__init__()
        self.host = host
        self.port = port
        self.background = torch.zeros(3, device="cuda")
        self.source_path = source_path
        self.gaussians = None
        self.mat = torch.eye(4).cuda()
        self.K = None
        self._keep_alive = False
        self._scaling = 1.0
    
    async def read_message(self, reader: asyncio.StreamReader):
        messageLength = await reader.readexactly(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = await reader.readexactly(messageLength)
        return json.loads(message.decode("utf-8"))
    
    async def send_message(self, writer: asyncio.StreamWriter, msg, verify):
        if msg is not None:
            msg = memoryview(msg)
            writer.write(msg)
            await writer.drain()
        writer.write(len(verify).to_bytes(4, "little"))
        writer.write(bytes(verify, "ascii"))
        await writer.drain()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                start = time.time()
                msg = await self.read_message(reader)
                width, height = msg["resolution_x"], msg["resolution_y"]

                if width > 0 and height > 0:
                    fovy, fovx, znear, zfar = msg["fov_y"], msg["fov_x"], msg["z_near"], msg["z_far"]
                    if self.K is not None:
                        fovx = focal2fov(self.K[0], width)
                        fovy = focal2fov(self.K[1], height)
                    # misuse sh_python to toggle depth
                    render_depth = bool(msg["shs_python"])
                    render_mask = bool(msg["rot_scale_python"])
                    self._keep_alive = bool(msg["keep_alive"])
                    scaling_modifier = msg["scaling_modifier"]

                    w2c = torch.reshape(torch.as_tensor(msg["view_matrix"]), (4, 4)).transpose(-1, -2).cuda()
                    w2c[1] *= -1
                    w2c[2] *= -1
                    # c2w = torch.linalg.inv(c2w)
                    # c2w[:, 3] = (c2w[:, 3] + self.translate) / self.radius
                    # w2c = torch.linalg.inv(c2w)
                    w2c = self.mat @ w2c
                    w2c[:3, 3] *= self._scaling
                    proj = getProjectionMatrix(znear, zfar, fovx, fovy).cuda()
                    cam = MiniCam(width, height, fovy, fovx, znear, zfar, w2c, proj)
                    image = self.render(cam, render_depth, render_mask, scaling_modifier)
                else:
                    image = None
                await self.send_message(writer, image, self.source_path)
                end = time.time()
                time.sleep(max(0, 1/60 - (end - start)))
        except Exception as e:
            print(e)

    @torch.no_grad()
    def render(self, cam: MiniCam, render_depth, render_mask, scaling_modifier=1.0):
        if self.gaussians is None:
            image = torch.randint(0, 255, (cam.image_height, cam.image_width, 3), dtype=torch.uint8).numpy()
        else:
            colors = torch.ones((len(self.gaussians), 3), device="cuda") if render_mask else None
            render_pkg = render(
                cam,
                self.gaussians,
                _pipe,
                self.background,
                scaling_modifier=scaling_modifier,
                override_color=colors
            )
            if render_depth:
                # arbitrary depth scale
                image = render_pkg["depth"].div(7)
            else:
                image = render_pkg["render"]
            if image.shape[0] == 1:
                image = image.repeat(3, 1 ,1)
            return torch.clamp(image, 0, 1).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

    @torch.no_grad()
    def set_gaussians(self, gaussians):
        self.gaussians = gaussians
    
    @torch.no_grad()
    def set_initial_pose(self, mat):
        self.mat = mat

    def set_intrinsics(self, K):
        self.K = K

    def set_scaling(self, s):
        self._scaling = s

    @property
    def keep_alive(self):
        return self._keep_alive

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        coroutine = asyncio.start_server(self.handle_client, self.host, self.port)
        server = loop.run_until_complete(coroutine)

        try:
            loop.run_forever()
        finally:
            server.close()
            loop.run_until_complete(server.wait_closed())
            loop.close()
