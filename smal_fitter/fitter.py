from smal_fitter.draw_smal_joints import SMALJointDrawer

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import reduce

from smal_fitter.p3d_renderer import Renderer
from smal_model.smal_torch import SMAL
from smal_fitter.utils import eul_to_axis

from smal_fitter.priors.joint_limits_prior import LimitPrior

import config


class Prior(object):
    """Pose prior sized from the model's J_names: zero mean (the rest pose), identity
    precision (independent angles), root rotation ignored."""

    def __init__(self, device):
        pose_len = len(config.dd["J_names"]) * 3

        self.precs = torch.from_numpy(np.identity(pose_len).copy()).float().to(device)
        self.mean = torch.from_numpy(np.zeros(pose_len)).float().to(device)

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:3] = False  # ignore rotation of base joint

        self.use_ind_tch = torch.from_numpy(self.use_ind).float().to(device)

    def __call__(self, x):
        mean_sub = x.reshape(-1, len(config.dd["J_names"]) * 3) - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims=([1], [0])) * self.use_ind_tch

        return res**2


class SMALFitter(nn.Module):
    def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior, rgb_only=False):
        super(SMALFitter, self).__init__()

        if use_unity_prior:
            raise ValueError(
                "use_unity_prior=True is not supported: the legacy Unity/WLDO quadruped prior was removed "
                "in #110. Pass use_unity_prior=False."
            )

        # for neuralSMIL we will only provide rgb images, no silhouettes or 2D keypoint data
        self.rgb_only = rgb_only

        if rgb_only:
            self.rgb_imgs = data_batch
            self.target_visibility = torch.ones(self.rgb_imgs.shape[0]).long()
            self.target_joints = torch.zeros(self.rgb_imgs.shape[0], config.N_POSE, 3)
        else:
            self.rgb_imgs, self.sil_imgs, self.target_joints, self.target_visibility = data_batch
            self.target_visibility = self.target_visibility.long()

        assert self.rgb_imgs.max() <= 1.0 and self.rgb_imgs.min() >= 0.0, "RGB Image range is incorrect"

        self.device = device
        self.num_images = self.rgb_imgs.shape[0]
        self.image_size = self.rgb_imgs.shape[2]

        self.batch_size = batch_size
        self.n_betas = config.N_BETAS

        self.shape_family_list = np.array(shape_family)

        self.propagate_scaling = False

        if config.DEBUG:
            print("Using shape prior learned from 3D scanned models")
        try:
            model_covs = config.dd["shape_cov"]
            self.mean_betas = torch.FloatTensor(config.dd["shape_mean_betas"])[: config.N_BETAS].to(device)
        except KeyError:
            if config.DEBUG:
                print(
                    "WARNING: shape_cov or shape_mean_betas not found in config.dd; "
                    "falling back to identity-covariance prior sized to config.N_BETAS"
                )
            model_covs = np.eye(config.N_BETAS)
            self.mean_betas = torch.zeros(config.N_BETAS).to(device)
        self.pose_prior = Prior(device)

        if config.DEBUG:
            print("\nShape covariance matrix")
            print(model_covs)
            print("\nShape mean betas")
            print(self.mean_betas, "\n")

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))  # why the addition? Avoiding zeroes?
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[: config.N_BETAS, : config.N_BETAS].to(device)

        self.betas = nn.Parameter(
            self.mean_betas.clone()
        )  # Shape parameters (1 for the entire sequence... note expand rather than repeat)

        # All bones/joints may scale independently
        self.log_beta_scales = torch.nn.Parameter(
            torch.zeros(self.num_images, config.N_POSE + 1, 3).to(device), requires_grad=False
        )
        # Per-joint translation parameters (disabled by default)
        self.betas_trans = torch.nn.Parameter(
            torch.zeros(self.num_images, config.N_POSE + 1, 3).to(device), requires_grad=False
        )

        # Per-joint rotation limits from the model .pkl ('joint_limits'), wide-open fallback;
        # treating everything as ball joints, see priors/joint_limits_prior.py
        try:
            limit_prior = LimitPrior()
            self._joint_limits_error = None
        except ValueError as e:
            # A malformed 'joint_limits' in the model file must not take
            # down consumers that never use the limit loss (e.g. neural
            # inference constructs a SMALFitter purely for rendering).
            # Defer the error to the first forward pass with w_limit > 0
            # and fall back to wide-open ranges meanwhile.
            self._joint_limits_error = e
            print(
                f"WARNING: invalid 'joint_limits' in the model file: {e} "
                "Using wide-open ranges; the limit loss (w_limit > 0) will raise."
            )
            _dd_orig = config.dd
            config.dd = {k: v for k, v in _dd_orig.items() if k != "joint_limits"}
            try:
                limit_prior = LimitPrior()
            finally:
                config.dd = _dd_orig
        # exclude root joint
        self.max_limits = torch.FloatTensor(limit_prior.max_values[3:]).view(config.N_POSE, 3).to(device)
        self.min_limits = torch.FloatTensor(limit_prior.min_values[3:]).view(config.N_POSE, 3).to(device)

        global_rotation_np = eul_to_axis(np.array([-np.pi / 2, 0, -np.pi / 2]))
        global_rotation = (
            torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(self.num_images, 1)
        )  # Global Init (Head-On)
        self.global_rotation = nn.Parameter(global_rotation)

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(self.num_images, 1)  # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(self.num_images, config.N_POSE, 3).to(device)
        self.joint_rotations = nn.Parameter(default_joints)

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0

        # Can be used to prevent certain joints rotating.
        # Can be useful depending on sequence.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device)
        # self.rotation_mask[25:32] = 0.0 # e.g. stop the tail moving

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(device, shape_family_id=shape_family)
        self.renderer = Renderer(self.image_size, device)
        # Adding the camera FOV as a tunable parameter
        self.fov = nn.Parameter(self.renderer.cameras.fov)

    def print_grads(self, grad_output):
        print(grad_output)

    def _batch_params(self, batch_range):
        """Model parameters for the frames in batch_range; generate_visualization writes these
        out to the per-frame .pkl files."""
        n_joints = self.joint_rotations.shape[1] + 1
        batch_params = {
            "global_rotation": self.global_rotation[batch_range] * self.global_mask,
            "joint_rotations": self.joint_rotations[batch_range] * self.rotation_mask,
            "betas": self.betas.expand(len(batch_range), self.n_betas),
            "trans": self.trans[batch_range],
            "fov": self.fov[batch_range],
            "log_betascale": self.log_beta_scales.expand(len(batch_range), n_joints, 3).to(self.device),
        }
        if hasattr(self, "betas_trans"):
            batch_params["betas_trans"] = self.betas_trans.expand(len(batch_range), n_joints, 3).to(self.device)
        return batch_params

    def forward(self, batch_range, weights, stage_id):
        # weights, here, correspond to the optimisation weighting set in the config file
        w_j2d, w_reproj, w_betas, w_pose, w_limit, w_splay = weights

        batch_params = self._batch_params(batch_range)

        target_joints = self.target_joints[batch_range].to(self.device)
        target_visibility = self.target_visibility[batch_range].to(self.device)
        if not self.rgb_only:
            sil_imgs = self.sil_imgs[batch_range].to(self.device)

        # Then, have the model retain its current 'appearance' taking as input the currently estimated model parameters
        # betas (shape input)
        # phi (global and model joint rotations)
        # optionally scale parameters
        verts, joints, Rs, v_shaped = self.smal_model(
            batch_params["betas"],
            torch.cat([batch_params["global_rotation"].unsqueeze(1), batch_params["joint_rotations"]], dim=1),
            betas_logscale=batch_params.get("log_betascale", None),
            betas_trans=batch_params.get("betas_trans", None),
            propagate_scaling=self.propagate_scaling,
        )

        verts = verts + batch_params["trans"].unsqueeze(1)
        joints = joints + batch_params["trans"].unsqueeze(1)

        canonical_model_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

        self.renderer.cameras.fov = self.fov
        rendered_silhouettes, rendered_joints = self.renderer(
            verts, canonical_model_joints, self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
        )

        objs = {}

        if w_j2d > 0:
            rendered_joints[~target_visibility.bool()] = -1.0
            target_joints[~target_visibility.bool()] = -1.0

            objs["joint"] = w_j2d * F.mse_loss(rendered_joints, target_joints)

        # Joint-limit hinge loss: flat inside [min, max], linear outside
        if w_limit > 0:
            if self._joint_limits_error is not None:
                raise ValueError(
                    "Limit loss is enabled (w_limit > 0) but the model's 'joint_limits' "
                    f"is invalid: {self._joint_limits_error}"
                )
            zeros = torch.zeros_like(batch_params["joint_rotations"])
            objs["limit"] = w_limit * torch.mean(
                torch.max(batch_params["joint_rotations"] - self.max_limits, zeros)
                + torch.max(self.min_limits - batch_params["joint_rotations"], zeros)
            )

        if w_pose > 0:
            objs["pose"] = (
                w_pose
                * self.pose_prior(
                    torch.cat([batch_params["global_rotation"].unsqueeze(1), batch_params["joint_rotations"]], dim=1)
                ).mean()
            )

        if w_splay > 0:
            objs["splay"] = w_splay * torch.sum(batch_params["joint_rotations"][:, :, [0, 2]] ** 2)

        if w_betas > 0:
            # TODO: add a regularizer on log_betascale; joint length scaling is currently unconstrained.
            diff_betas = batch_params["betas"] - self.mean_betas.unsqueeze(0)  # N, B
            res = torch.tensordot(diff_betas, self.betas_prec, dims=([1], [0]))
            objs["betas"] = w_betas * (res**2).mean()

        if w_reproj > 0 and not self.rgb_only:
            objs["sil_reproj"] = w_reproj * F.l1_loss(rendered_silhouettes, sil_imgs)

        return reduce(lambda x, y: x + y, objs.values()), objs

    def get_temporal(self, w_temp):
        joint_rotations = self.joint_rotations * self.rotation_mask
        global_rotation = self.global_rotation * self.global_mask

        joint_loss = torch.tensor(0.0).float().to(self.device)
        global_loss = torch.tensor(0.0).float().to(self.device)
        trans_loss = torch.tensor(0.0).float().to(self.device)

        for i in range(0, self.num_images - 1):
            global_loss += F.mse_loss(global_rotation[i], global_rotation[i + 1]) * w_temp
            joint_loss += F.mse_loss(joint_rotations[i], joint_rotations[i + 1]) * w_temp
            trans_loss += F.mse_loss(self.trans[i], self.trans[i + 1]) * w_temp

        return joint_loss, global_loss, trans_loss

    def generate_visualization(self, image_exporter, apply_UE_transform=False, img_idx=0, mesh_scale=None, epoch=None):
        """
        Generate visualization images with rendered mesh overlay.

        Args:
            image_exporter: Image exporter instance for saving outputs
            apply_UE_transform: If True, apply 10x scaling (Unreal Engine convention)
            img_idx: Image index for naming
            mesh_scale: Optional mesh scale factor (float or tensor). If provided,
                       mesh is centered at root joint, scaled, then translated.
                       This should match the scale used in training when
                       allow_mesh_scaling is enabled.
            epoch: Current epoch number for naming visualization images
        """
        for j in range(0, self.num_images, self.batch_size):
            batch_range = list(range(j, min(self.num_images, j + self.batch_size)))
            batch_params = self._batch_params(batch_range)

            target_joints = self.target_joints[batch_range]
            target_visibility = self.target_visibility[batch_range]
            rgb_imgs = self.rgb_imgs[batch_range].to(self.device)
            if not self.rgb_only:
                sil_imgs = self.sil_imgs[batch_range].to(self.device)
            else:
                # Create dummy silhouette for rgb_only mode
                sil_imgs = torch.zeros_like(rgb_imgs[:, :1, :, :])

            with torch.no_grad():
                verts, joints, Rs, v_shaped = self.smal_model(
                    batch_params["betas"],
                    torch.cat([batch_params["global_rotation"].unsqueeze(1), batch_params["joint_rotations"]], dim=1),
                    betas_logscale=batch_params.get("log_betascale", None),
                    betas_trans=batch_params.get("betas_trans", None),
                    propagate_scaling=self.propagate_scaling,
                )

                if apply_UE_transform:
                    # in UE5 the model is scaled up by 10 (double check model size in your replicant project, if modified)
                    # needed to align the model at the root joint and scale it to the replicAnt model size
                    verts = (verts - joints[:, 0, :]) * 10 + batch_params["trans"].unsqueeze(1)
                    joints = (joints - joints[:, 0, :]) * 10 + batch_params["trans"].unsqueeze(1)
                elif mesh_scale is not None:
                    # Apply mesh scaling (used when allow_mesh_scaling is enabled in training)
                    # Center at root joint, scale, then translate - matches _render_keypoints_with_camera
                    scale_val = mesh_scale
                    if isinstance(scale_val, (int, float)):
                        scale_val = torch.tensor([[scale_val]], device=self.device, dtype=torch.float32)
                    else:
                        # Ensure tensor is float32
                        scale_val = scale_val.to(device=self.device, dtype=torch.float32)
                        if scale_val.dim() == 0:
                            scale_val = scale_val.unsqueeze(0).unsqueeze(0)
                        elif scale_val.dim() == 1:
                            scale_val = scale_val.unsqueeze(1)
                    root_joint = joints[:, 0:1, :]
                    verts = (verts - root_joint) * scale_val.unsqueeze(-1) + batch_params["trans"].unsqueeze(1)
                    joints = (joints - root_joint) * scale_val.unsqueeze(-1) + batch_params["trans"].unsqueeze(1)
                else:
                    verts = verts + batch_params["trans"].unsqueeze(1)
                    joints = joints + batch_params["trans"].unsqueeze(1)

                canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

                # Ensure float32 for rendering (SMAL model can output float64 from some buffers)
                verts = verts.float()
                canonical_joints = canonical_joints.float()

                rendered_silhouettes, rendered_joints, rendered_images = self.renderer(
                    verts,
                    canonical_joints,
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1),
                    render_texture=True,
                )

                # Second view: the camera orbiting the subject by 90 degrees about its x axis,
                # framed to fill the panel. Placed in camera space, so it works for both
                # camera-centric and model-centric checkpoints (see Renderer.orbit_view).
                orbit_verts, orbit_joints = self.renderer.orbit_view(verts, canonical_joints)
                _, rev_joints, rev_images = self.renderer(
                    orbit_verts,
                    orbit_joints,
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1),
                    render_texture=True,
                )

                overlay_image = (rendered_images * 0.5) + (rgb_imgs * 0.5)

                # Get image size for resolution-aware keypoint scaling
                image_size = rgb_imgs.shape[-1]  # Assuming square images, get height/width

                target_vis = SMALJointDrawer.draw_joints(
                    rgb_imgs, target_joints, visible=target_visibility, image_size=image_size
                )
                rendered_images_vis = SMALJointDrawer.draw_joints(
                    rendered_images, rendered_joints, visible=target_visibility, image_size=image_size
                )
                rendered_overlay_vis = SMALJointDrawer.draw_joints(
                    overlay_image, rendered_joints, visible=target_visibility, image_size=image_size
                )
                rev_images_vis = SMALJointDrawer.draw_joints(
                    rev_images, rev_joints, visible=target_visibility, image_size=image_size
                )

                silhouette_error = 1.0 - F.l1_loss(sil_imgs, rendered_silhouettes, reduction="none")
                silhouette_error = silhouette_error.expand_as(rgb_imgs).data.cpu()

                collage_rows = torch.cat(
                    [target_vis, rendered_images_vis, rendered_overlay_vis, silhouette_error, rev_images_vis], dim=3
                )

                for batch_id, global_id in enumerate(batch_range):
                    collage_np = np.transpose(collage_rows[batch_id].numpy(), (1, 2, 0))
                    img_parameters = {k: v[batch_id].cpu().data.numpy() for (k, v) in batch_params.items()}
                    image_exporter.export(
                        (collage_np * 255.0).astype(np.uint8),
                        batch_id,
                        global_id,
                        img_parameters,
                        verts,
                        self.smal_model.faces.data.cpu().numpy(),
                        img_idx,
                        epoch=epoch,
                    )
