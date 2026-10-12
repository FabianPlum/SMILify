# Data structures and functions for rendering
import torch
import numpy as np
import config

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    PointLights,
    HardPhongShader,
    SoftSilhouetteShader,
    Textures,
    FoVPerspectiveCameras,
)


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
        # self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T, fov=60)
        self.cameras = FoVPerspectiveCameras(
            R=R, T=T, device=device, fov=60, znear=self.DEFAULT_ZNEAR, zfar=self.DEFAULT_ZFAR
        )
        self.mesh_color = torch.FloatTensor(config.MESH_COLOR).to(device)[None, None, :] / 255.0

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=100,
            bin_size=0,
        )

        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

        raster_settings_color = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,  # Use naive rasterization to avoid bin overflow with large/distant meshes
        )

        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        self.color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings_color),
            shader=HardPhongShader(
                device=device,
                cameras=self.cameras,
                lights=lights,
            ),
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
            device=self.device,
            R=R,
            T=T,
            fov=fov,
            aspect_ratio=aspect_ratio,
            znear=self.DEFAULT_ZNEAR,
            zfar=self.DEFAULT_ZFAR,
        )
        self.silhouette_renderer.rasterizer.cameras = self.cameras
        self.color_renderer.rasterizer.cameras = self.cameras
        # Also update shader cameras if they have a cameras attribute
        if hasattr(self.color_renderer.shader, "cameras"):
            self.color_renderer.shader.cameras = self.cameras

    def orbit_view(self, vertices, points, angle_deg=90.0, axis="camera_x", fill=0.8):
        """Place the subject for a second view: the camera orbiting it by *angle_deg* about *axis*.

        Works in camera (view) space, so it does not matter where the camera sits in the
        world (camera-centric identity camera or model-centric predicted camera). The
        subject is rotated about its own centre, then put on the optical axis at the
        distance where its bounding sphere fills *fill* of the narrower half field of
        view, so it is always fully in front of the camera and in frame.

        Args:
            vertices, points: (B, N, 3) world-space mesh vertices and joints.
            angle_deg: Orbit angle.
            axis: "camera_x" (horizontal image axis; 90 deg gives a view over the top),
                "camera_y" (vertical image axis), or "world_y" (the world's y axis; with
                180 deg this is the old turn-around view for model-centric checkpoints).
            fill: Fraction of the narrower half field of view the bounding sphere spans.

        Returns:
            World-space (vertices, points) to render with the unchanged camera.
        """
        w2v = self.cameras.get_world_to_view_transform()
        v = w2v.transform_points(vertices.float())
        p = w2v.transform_points(points.float())
        centre = v.mean(dim=1, keepdim=True)

        batch = v.shape[0]
        if axis == "camera_x":
            k = torch.tensor([1.0, 0.0, 0.0], device=v.device).expand(batch, 3)
        elif axis == "camera_y":
            k = torch.tensor([0.0, 1.0, 0.0], device=v.device).expand(batch, 3)
        elif axis == "world_y":
            ends = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=v.device).expand(batch, 2, 3)
            ends = w2v.transform_points(ends)
            k = ends[:, 1] - ends[:, 0]
        else:
            raise ValueError(f"Unknown orbit axis '{axis}'; use 'camera_x', 'camera_y' or 'world_y'.")
        k = k / k.norm(dim=-1, keepdim=True)

        # Rodrigues: R = I cos(a) + sin(a) [k]x + (1 - cos(a)) k k^T, per sample
        a = torch.tensor(np.radians(angle_deg), dtype=v.dtype, device=v.device)
        zero = torch.zeros(batch, device=v.device)
        k_cross = torch.stack(
            [
                torch.stack([zero, -k[:, 2], k[:, 1]], dim=-1),
                torch.stack([k[:, 2], zero, -k[:, 0]], dim=-1),
                torch.stack([-k[:, 1], k[:, 0], zero], dim=-1),
            ],
            dim=1,
        )
        eye = torch.eye(3, device=v.device).expand(batch, 3, 3)
        rot = eye * torch.cos(a) + torch.sin(a) * k_cross + (1 - torch.cos(a)) * k.unsqueeze(2) * k.unsqueeze(1)

        v = (v - centre) @ rot.transpose(1, 2)
        p = (p - centre) @ rot.transpose(1, 2)

        radius = v.norm(dim=-1).max(dim=1).values
        half_fov = torch.deg2rad(self.cameras.fov.reshape(-1).float()) / 2  # vertical
        aspect = self.cameras.aspect_ratio.reshape(-1).float()
        half_fov = torch.atan(torch.tan(half_fov) * aspect.clamp(max=1.0))  # narrower of vertical / horizontal
        offset = torch.zeros_like(centre)
        offset[:, 0, 2] = radius / torch.sin(half_fov * fill)  # nearest point at distance - radius > 0

        v2w = w2v.inverse()
        return v2w.transform_points(v + offset), v2w.transform_points(p + offset)

    def forward(self, vertices, points, faces, render_texture=False, joints_only=False):
        # PyTorch3D expects float32 inputs; upstream code can accidentally produce float64
        # (e.g., from SMAL model buffers / numpy defaults). Force here to avoid dtype mismatch.
        vertices = vertices.float()
        points = points.float()
        faces = faces.long()

        screen_size = torch.ones(vertices.shape[0], 2, dtype=torch.float32, device=vertices.device) * self.image_size
        # fix discussed here: https://github.com/benjiebob/SMALify/issues/30
        # answer by mshooter
        proj_points = self.cameras.transform_points_screen(points, image_size=screen_size)[:, :, [1, 0]]

        if joints_only:
            return None, proj_points

        tex = torch.ones_like(vertices) * self.mesh_color  # (1, V, 3)
        textures = Textures(verts_rgb=tex)

        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)

        if render_texture:
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            return sil_images, proj_points
