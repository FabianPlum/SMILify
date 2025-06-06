# Data structures and functions for rendering
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
import config
import cv2

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, HardPhongShader, SoftSilhouetteShader, Materials, Textures,
    FoVPerspectiveCameras
)
from pytorch3d.io import load_objs_as_meshes
from utils import perspective_proj_withz


class Renderer(torch.nn.Module):
    def __init__(self, image_size, device):
        super(Renderer, self).__init__()

        self.image_size = image_size
        self.device = device

        # Initialize with default camera parameters
        R, T = look_at_view_transform(2.7, 0, 0, device=device)
        #self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T, fov=60)
        self.cameras = FoVPerspectiveCameras(R=R, T=T, device=device, fov=60)
        print("Camera field of view:", str(self.cameras.fov.item()), "degrees")
        self.mesh_color = torch.FloatTensor(config.MESH_COLOR).to(device)[None, None, :] / 255.0

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100, bin_size=0
        )

        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        raster_settings_color = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        self.color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings_color
            ),
            shader=HardPhongShader(
                device=device,
                cameras=self.cameras,
                lights=lights,
            )
        )

    def set_camera_parameters(self, R, T, fov):
        """
        Set the camera parameters for the renderer.

        Args:
            R (torch.Tensor): Rotation matrix of the camera.
            T (torch.Tensor): Translation vector of the camera.
            fov (torch.Tensor): Field of view of the camera.
        """
        self.cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        self.silhouette_renderer.rasterizer.cameras = self.cameras
        self.color_renderer.rasterizer.cameras = self.cameras

    def forward(self, vertices, points, faces, render_texture=False):
        tex = torch.ones_like(vertices) * self.mesh_color  # (1, V, 3)
        textures = Textures(verts_rgb=tex)

        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)
        screen_size = torch.ones(vertices.shape[0], 2).to(vertices.device) * self.image_size
        # fix discussed here: https://github.com/benjiebob/SMALify/issues/30
        # answer by mshooter
        proj_points = self.cameras.transform_points_screen(points, image_size=screen_size)[:, :, [1, 0]]

        if render_texture:
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            return sil_images, proj_points
