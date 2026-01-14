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
    # Clipping plane defaults - wide range to handle various scene scales
    # (SLEAP data can be in meters with objects at 0.1-10m from camera)
    DEFAULT_ZNEAR = 0.001  # Very close (1mm)
    DEFAULT_ZFAR = 1000.0  # Very far (1km)
    
    def __init__(self, image_size, device):
        super(Renderer, self).__init__()

        self.image_size = image_size
        self.device = device

        # Initialize with default camera parameters
        R, T = look_at_view_transform(2.7, 0, 0, device=device)
        #self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T, fov=60)
        self.cameras = FoVPerspectiveCameras(
            R=R, T=T, device=device, fov=60,
            znear=self.DEFAULT_ZNEAR, zfar=self.DEFAULT_ZFAR
        )
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
            bin_size=0,  # Use naive rasterization to avoid bin overflow with large/distant meshes
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

    def set_camera_parameters(self, R, T, fov, aspect_ratio=None):
        """
        Set the camera parameters for the renderer.

        Args:
            R (torch.Tensor): Rotation matrix of the camera.
            T (torch.Tensor): Translation vector of the camera.
            fov (torch.Tensor): Field of view of the camera.
            aspect_ratio (Optional[torch.Tensor|float]): Aspect ratio for FoVPerspectiveCameras.
                If provided, this is used to correctly handle non-square camera intrinsics
                (e.g. SLEAP/OpenCV calibrations with W!=H and/or fx!=fy).
        """
        # Ensure all camera params are float32 on the correct device
        R = R.to(device=self.device, dtype=torch.float32)
        T = T.to(device=self.device, dtype=torch.float32)
        fov = fov.to(device=self.device, dtype=torch.float32)
        
        # Ensure FOV has the correct shape for FoVPerspectiveCameras
        # FoVPerspectiveCameras expects fov to be 1D tensor of shape (batch_size,)
        if fov.dim() > 1:
            fov = fov.squeeze(-1)  # Remove trailing dimensions
        if fov.dim() == 0:
            fov = fov.unsqueeze(0)  # Add batch dimension if scalar

        # FoVPerspectiveCameras cannot handle aspect_ratio=None (it is used in arithmetic).
        # Default to 1.0 (square pixels / symmetric intrinsics) when not provided.
        if aspect_ratio is None:
            aspect_ratio = torch.ones_like(fov, dtype=torch.float32, device=self.device)
        else:
            if not isinstance(aspect_ratio, torch.Tensor):
                aspect_ratio = torch.tensor(aspect_ratio, dtype=torch.float32, device=self.device)
            aspect_ratio = aspect_ratio.to(device=self.device, dtype=torch.float32)
            if aspect_ratio.dim() > 1:
                aspect_ratio = aspect_ratio.squeeze(-1)
            if aspect_ratio.dim() == 0:
                aspect_ratio = aspect_ratio.unsqueeze(0)
            # Broadcast scalar aspect ratio to batch if needed
            if aspect_ratio.numel() == 1 and fov.numel() > 1:
                aspect_ratio = aspect_ratio.expand_as(fov)
        
        self.cameras = FoVPerspectiveCameras(
            device=self.device, R=R, T=T, fov=fov, aspect_ratio=aspect_ratio,
            znear=self.DEFAULT_ZNEAR, zfar=self.DEFAULT_ZFAR
        )
        self.silhouette_renderer.rasterizer.cameras = self.cameras
        self.color_renderer.rasterizer.cameras = self.cameras
        # Also update shader cameras if they have a cameras attribute
        if hasattr(self.color_renderer.shader, 'cameras'):
            self.color_renderer.shader.cameras = self.cameras

    def forward(self, vertices, points, faces, render_texture=False):
        # PyTorch3D expects float32 inputs; upstream code can accidentally produce float64
        # (e.g., from SMAL model buffers / numpy defaults). Force here to avoid dtype mismatch.
        vertices = vertices.float()
        points = points.float()
        faces = faces.long()

        tex = torch.ones_like(vertices) * self.mesh_color  # (1, V, 3)
        textures = Textures(verts_rgb=tex)

        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)
        screen_size = torch.ones(vertices.shape[0], 2, dtype=torch.float32, device=vertices.device) * self.image_size
        # fix discussed here: https://github.com/benjiebob/SMALify/issues/30
        # answer by mshooter
        proj_points = self.cameras.transform_points_screen(points, image_size=screen_size)[:, :, [1, 0]]

        if render_texture:
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            return sil_images, proj_points
