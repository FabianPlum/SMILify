"""Introduces Stage class - representing a Stage of optimising a batch of SMBLD meshes to target meshes"""

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from fitter_3d.utils import plot_meshes, SDF_distance, sample_points_from_meshes_and_SDF, try_mkdir
import numpy as np
import os
import config
import pickle as pkl

from smal_model.smal_torch import SMAL
from smal_fitter.utils import eul_to_axis
from smal_fitter.priors.joint_limits_prior import _ranges_from_joint_limits

nn = torch.nn

default_weights = dict(
    w_chamfer=1.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, w_sdf=0.5, w_limit=0.0
)  # Added SDF distance weight; w_limit off by default (issue #97)
# Want to vary learning ratios between parameters,
default_lr_ratios = []


def get_meshes(verts, faces, device="cuda"):
    """Returns Meshes object of all SMAL meshes."""
    meshes = Meshes(verts=verts, faces=faces).to(device)
    return meshes


def _joint_limit_tensors_from_dd(dd, device):
    """Build (min_limits, max_limits, error) tensors of shape (N_POSE, 3) from an
    already-loaded SMAL .pkl dict. `error` is the deferred ValueError/TypeError (or
    None) if 'joint_limits' was present but malformed - validation is never raised
    here, only reported, so construction never crashes a consumer that doesn't use
    w_limit."""

    def _safe_ranges(dd):
        try:
            return _ranges_from_joint_limits(dd), None
        except (ValueError, TypeError) as e:
            # Fall back to wide-open ranges. Build the fallback from a *copy*
            # of dd with 'joint_limits' stripped - dd itself is never mutated.
            dd_no_limits = {k: v for k, v in dd.items() if k != "joint_limits"}
            return _ranges_from_joint_limits(dd_no_limits), e

    ranges, error = _safe_ranges(dd)
    if error is not None:
        print(
            f"WARNING: invalid 'joint_limits' in the model file: {error} "
            "Using wide-open ranges; the limit loss (w_limit > 0) will raise."
        )

    joint_names = dd["J_names"]
    n_pose = len(joint_names) - 1  # not including the root/global rotation
    min_values = np.array([ranges[j][ax][0] for j in joint_names for ax in range(3)])
    max_values = np.array([ranges[j][ax][1] for j in joint_names for ax in range(3)])

    # exclude root joint (its 3 values come first), reshape to match joint_rot
    min_limits = torch.FloatTensor(min_values[3:]).view(n_pose, 3).to(device)
    max_limits = torch.FloatTensor(max_values[3:]).view(n_pose, 3).to(device)
    return min_limits, max_limits, error


class SMAL3DFitter(nn.Module):
    def __init__(self, batch_size=1, device="cuda", shape_family=-1):
        super(SMAL3DFitter, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_betas = config.N_BETAS

        with open(config.SMAL_FILE, "rb") as f:
            u = pkl._Unpickler(f)
            u.encoding = "latin1"
            dd = u.load()

        try:
            model_covs = dd["shape_cov"]
            self.mean_betas = torch.FloatTensor(dd["shape_mean_betas"]).to(device)
        except KeyError:
            print(
                "No shape_cov or shape_mean_betas found in SMAL_FILE; "
                "falling back to identity-covariance prior sized to config.N_BETAS"
            )
            self.mean_betas = torch.zeros(config.N_BETAS).to(device)
            model_covs = np.eye(config.N_BETAS)

        if config.DEBUG:
            print("MODEL COVS", model_covs)

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[: config.N_BETAS, : config.N_BETAS].to(device)

        if config.DEBUG:
            print("MEAN BETAS", self.mean_betas)

        self.betas = nn.Parameter(self.mean_betas.unsqueeze(0).repeat(batch_size, 1))

        # Reuses the dd loaded above rather than re-opening config.SMAL_FILE.
        self.kintree_table = torch.tensor(dd["kintree_table"]).to(device)

        # Get number of joints from kintree
        self.n_joints = self.kintree_table.shape[1]

        # Initialize log_beta_scales with proper shape: batch_size x n_joints x 3
        # Starting with ones means no scaling initially (since exp(0) = 1)
        self.log_beta_scales = nn.Parameter(torch.zeros(self.batch_size, self.n_joints, 3).to(device))

        # Initialize betas_trans with proper shape: batch_size x n_joints x 3
        # Starting with zeros means no translation offsets initially
        self.betas_trans = nn.Parameter(torch.zeros(self.batch_size, self.n_joints, 3).to(device))

        global_rotation_np = eul_to_axis(np.array([0, 0, 0]))
        global_rotation = (
            torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(batch_size, 1)
        )  # Global Init (Head-On)
        self.global_rot = nn.Parameter(global_rotation)

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(batch_size, 1)  # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(batch_size, config.N_POSE, 3).to(device)
        self.joint_rot = nn.Parameter(default_joints)

        # Joint rotation limits (issue #97): read authored 'joint_limits' from the
        # model .pkl and build a hinge-loss prior, mirroring smal_fitter/fitter.py's
        # SMALFitter.__init__. Built from the local `dd` loaded above (not
        # LimitPrior()/config.dd) so this doesn't depend on config.dd staying in sync
        # with config.SMAL_FILE for this class.
        #
        # A malformed 'joint_limits' in the model file must not take down every run
        # that constructs a SMAL3DFitter (this class is built unconditionally, before
        # any stage's loss_weights are even looked at - see optimise.py). So we defer
        # the error to the first forward pass with w_limit > 0 (Stage.forward()) and
        # fall back to wide-open ranges meanwhile.
        self.min_limits, self.max_limits, self._joint_limits_error = _joint_limit_tensors_from_dd(dd, device)

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0

        # Can be used to prevent certain joints rotating.
        # Can be useful depending on sequence.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device)  # by default all joints are free to rotate
        # self.rotation_mask[25:32] = 0.0 # e.g. stop the tail moving

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(device, shape_family_id=shape_family)
        self.faces = self.smal_model.faces.unsqueeze(0).repeat(batch_size, 1, 1)

        """
        # vertex offsets for deformations
        self.deform_verts = nn.Parameter(torch.zeros(batch_size, *self.smal_model.v_template.shape)).to(device)
        """

        # Initialize deform_verts as a leaf tensor
        self.deform_verts = nn.Parameter(
            torch.zeros(batch_size, *self.smal_model.v_template.shape).to(device), requires_grad=True
        )

    def get_joint_scales(self, log_beta_scales_arg=None):
        """
        Compute the final scale for each joint taking into account the kinematic chain.
        A joint's scale is influenced by its own scale parameters and all its parents.

        Args:
            log_beta_scales_arg (optional): Tensor of shape (batch_size, n_joints, 3)
                                            to use instead of self.log_beta_scales.
        """
        current_log_beta_scales = log_beta_scales_arg if log_beta_scales_arg is not None else self.log_beta_scales
        # Start with base scales from log_beta_scales
        joint_scales = torch.exp(current_log_beta_scales)  # Convert from log space

        # For each joint
        for joint_idx in range(self.n_joints):
            parent_idx = self.kintree_table[0, joint_idx]

            # If this joint has a parent (parent_idx != joint_idx)
            # and the parent_idx is valid (within bounds of joint_scales)
            if parent_idx != joint_idx and parent_idx < joint_scales.shape[1]:
                # Accumulate parent's scale
                joint_scales[:, joint_idx] = joint_scales[:, joint_idx] * joint_scales[:, parent_idx]

        return joint_scales

    def forward(
        self,
        betas=None,
        global_rot=None,
        joint_rot=None,
        trans=None,
        log_beta_scales=None,
        betas_trans=None,
        deform_verts=None,
        return_joints=False,
    ):
        """
        Forward pass for the SMAL model.
        Can accept optional parameters to override the internal nn.Parameter attributes.

        Args:
            betas (optional): Shape parameters, tensor of shape (batch_size, N_BETAS).
            global_rot (optional): Global rotation in axis-angle, tensor of shape (batch_size, 3).
            joint_rot (optional): Joint rotations in axis-angle, tensor of shape (batch_size, N_POSE, 3).
            trans (optional): Global translation, tensor of shape (batch_size, 3).
            log_beta_scales (optional): Logarithm of joint scales, tensor of shape (batch_size, n_joints, 3).
            betas_trans (optional): Joint translation offsets, tensor of shape (batch_size, n_joints, 3).
            deform_verts (optional): Vertex offsets, tensor of shape (batch_size, n_template_verts, 3).
            return_joints (bool): Whether to return joints.

        Returns:
            verts: Predicted vertices, tensor of shape (batch_size, n_verts, 3).
            joints: Predicted joints, tensor of shape (batch_size, n_joints, 3), if return_joints is True.
        """

        # Determine which parameters to use (passed argument or self.attribute)
        _betas = betas if betas is not None else self.betas
        _global_rot = global_rot if global_rot is not None else self.global_rot
        _joint_rot = joint_rot if joint_rot is not None else self.joint_rot
        _trans = trans if trans is not None else self.trans
        # Use provided log_beta_scales if available, otherwise use the internal one.
        # This will be passed to get_joint_scales and directly to smal_model if needed.
        _log_beta_scales_to_use = log_beta_scales if log_beta_scales is not None else self.log_beta_scales
        _betas_trans = betas_trans if betas_trans is not None else self.betas_trans
        _deform_verts = deform_verts if deform_verts is not None else self.deform_verts

        # The original get_joint_scales uses self.log_beta_scales.
        # We don't need joint_scales for the smal_model call directly,
        # as smal_model itself handles the betas_logscale.
        # The get_joint_scales method itself is not directly used in the smal_model call here,
        # but it's good practice to make it consistent if it were to be used externally
        # or if smal_model's interface changes.
        # For the current self.smal_model call, we pass _log_beta_scales_to_use directly.

        # The SMAL model expects betas_logscale to be of shape (batch_size, n_joints, 3).
        # self.log_beta_scales is already initialized with this shape.
        # If log_beta_scales is passed as an argument, it should also conform to this shape.
        # The reshape operation in the original code was:
        # `betas_logscale = self.log_beta_scales.reshape(self.batch_size, -1, 3)`
        # This is effectively a no-op if self.log_beta_scales is already (bs, n_joints, 3).

        verts, joints, Rs, v_shaped = self.smal_model(
            _betas,
            torch.cat(
                [
                    _global_rot.unsqueeze(1),  # _global_rot is (bs, 3) -> (bs, 1, 3)
                    _joint_rot,
                ],
                dim=1,
            ),  # _joint_rot is (bs, N_POSE, 3)
            betas_logscale=_log_beta_scales_to_use,  # Pass the determined log_beta_scales
            betas_trans=_betas_trans,
        )  # Pass the joint translation offsets

        verts = verts + _trans.unsqueeze(1)  # _trans is (bs, 3) -> (bs, 1, 3)
        joints = joints + _trans.unsqueeze(1)

        verts = verts + _deform_verts  # _deform_verts is (bs, n_template_verts, 3)

        if return_joints:
            return verts, joints
        else:
            return verts


class SMALParamGroup:
    """Object building on model.parameters, with modifications such as variable learning rate"""

    param_map = {
        "init": ["global_rot", "trans"],
        "init_rot_lock": ["trans", "log_beta_scales"],
        "init_rot_lock_trans": ["trans", "betas_trans"],
        "init_rot_lock_trans_scale": ["trans", "betas_trans", "log_beta_scales"],
        "default": ["global_rot", "joint_rot", "trans", "betas", "log_beta_scales"],
        "default_with_betas_trans": ["global_rot", "joint_rot", "trans", "betas", "log_beta_scales", "betas_trans"],
        "shape": ["global_rot", "trans", "betas", "log_beta_scales", "betas_trans"],
        "pose": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "betas_trans"],
        "deform": ["deform_verts"],
        "all": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "betas_trans", "deform_verts"],
    }  # map of param_type : all attributes in SMAL used in optim

    def __init__(self, model, group="smbld", lrs=None):
        """
        :param lrs: dict of param_name : custom learning rate
        """

        self.model = model

        self.group = group
        assert group in self.param_map, f"Group {group} not in list of available params: {list(self.param_map.keys())}"

        self.lrs = {}
        if lrs is not None:
            for k, lr in lrs.items():
                self.lrs[k] = lr

    def __iter__(self):
        """Return iterable list of all parameters"""
        out = []

        for param_name in self.param_map[self.group]:
            param = [getattr(self.model, param_name)]
            d = {"params": param}
            if param_name in self.lrs:
                d["lr"] = self.lrs[param_name]

            out.append(d)

        return iter(out)


class Stage:
    """Defines a stage of optimisation, the optimisation parameters for the stage, ..."""

    def __init__(
        self,
        nits: int,
        scheme: str,
        smal_3d_fitter: SMAL3DFitter,
        target_meshes: Meshes,
        mesh_names=[],
        name="optimise",
        loss_weights=None,
        lr=1e-3,
        out_dir="static_fits_output",
        custom_lrs=None,
        device="cuda",
        plot_normals=False,
        sample_size=1000,
        sdf_values=None,
        source_sdf_values=None,
        visualize_sdf_loss=False,
        sdf_vis_frequency=10,
    ):
        """
        nits = integer, number of iterations in stage
        parameters = list of items over which to be optimised
        get_mesh = function that returns Mesh object for identifying losses
        name = name of stage
        visualize_sdf_loss = whether to visualize SDF loss contribution
        sdf_vis_frequency = how often to visualize SDF loss (every N iterations)

        lr_decay = factor by which lr decreases at each it"""

        self.n_it = nits
        self.name = name
        self.out_dir = out_dir
        self.target_meshes = target_meshes
        self.mesh_names = mesh_names
        self.smal_3d_fitter = smal_3d_fitter
        self.device = device
        self.plot_normals = plot_normals
        self.loss_weights = default_weights.copy()
        if loss_weights is not None:
            for k, v in loss_weights.items():
                self.loss_weights[k] = v

        # Parameter for vertex sampling
        self.sample_size = sample_size

        # Store SDF values if provided
        self.sdf_values = sdf_values
        self.source_sdf_values = source_sdf_values

        # SDF loss visualization parameters
        self.visualize_sdf_loss = visualize_sdf_loss
        self.sdf_vis_frequency = sdf_vis_frequency

        self.losses_to_plot = []  # Store losses for review later

        if custom_lrs is not None:
            for attr in custom_lrs:
                assert hasattr(smal_3d_fitter, attr), f"attr '{attr}' not in SMAL."

        self.param_group = SMALParamGroup(smal_3d_fitter, scheme, custom_lrs)

        self.scheduler = None

        self.optimizer = torch.optim.Adam(self.param_group, lr=lr)
        self.src_verts = smal_3d_fitter().detach()  # original verts, detach from autograd
        self.faces = smal_3d_fitter.faces.detach()
        self.src_mesh = get_meshes(self.src_verts, self.faces, device=device)
        self.n_verts = self.src_verts.shape[1]

        self.consider_loss = lambda loss_name: (
            self.loss_weights[f"w_{loss_name}"] > 0
        )  # function to check if loss is non-zero

    def forward(self, src_mesh, iteration=0):
        loss = 0
        loss_components = {}

        # Sample from target meshes
        target_verts = sample_points_from_meshes(self.target_meshes, 3000)

        if self.consider_loss("chamfer"):
            loss_chamfer, _ = chamfer_distance(target_verts, src_mesh.verts_padded())
            loss_components["chamfer"] = loss_chamfer
            loss += self.loss_weights["w_chamfer"] * loss_chamfer

        if self.consider_loss("edge"):
            loss_edge = mesh_edge_loss(src_mesh)  # and (b) the edge length of the predicted mesh
            loss_components["edge"] = loss_edge
            loss += self.loss_weights["w_edge"] * loss_edge

        if self.consider_loss("normal"):
            loss_normal = mesh_normal_consistency(src_mesh)  # mesh normal consistency
            loss_components["normal"] = loss_normal
            loss += self.loss_weights["w_normal"] * loss_normal

        if self.consider_loss("laplacian"):
            loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")  # mesh laplacian smoothing
            loss_components["laplacian"] = loss_laplacian
            loss += self.loss_weights["w_laplacian"] * loss_laplacian

        # Add SDF distance loss if SDF values are provided
        if self.consider_loss("sdf") and self.sdf_values is not None and self.source_sdf_values is not None:
            # Sample points from source mesh for SDF calculation
            src_verts, src_sdf = sample_points_from_meshes_and_SDF(src_mesh, self.source_sdf_values, 10000)
            target_verts, target_sdf = sample_points_from_meshes_and_SDF(self.target_meshes, self.sdf_values, 10000)

            # Determine if we should visualize on this iteration
            visualize_now = self.visualize_sdf_loss and (
                iteration % self.sdf_vis_frequency == 0 or iteration == self.n_it - 1
            )

            # Create visualization directory if needed
            if visualize_now:
                vis_dir = os.path.join(self.out_dir, "sdf_visualization", self.name)
                try_mkdir(vis_dir)
            else:
                vis_dir = "sdf_visualization"

            # Calculate SDF distance
            loss_sdf = SDF_distance(
                src_verts,  # source points
                target_verts,  # target points
                src_sdf,  # source SDF values
                target_sdf,  # target SDF values
                k=50,  # number of nearest neighbors
                batch_reduction="mean",
                point_reduction="mean",
                norm=2,
                single_directional=False,
                visualize=visualize_now,
                output_dir=vis_dir,
                title=f"{self.name}_iteration{iteration}",
                mesh_names=self.mesh_names,  # Pass mesh names for better file naming
            )
            loss_components["sdf"] = loss_sdf
            loss += self.loss_weights["w_sdf"] * loss_sdf

        if self.consider_loss("limit"):
            if self.smal_3d_fitter._joint_limits_error is not None:
                raise ValueError(
                    "Limit loss is enabled (w_limit > 0) but the model's 'joint_limits' "
                    f"is invalid: {self.smal_3d_fitter._joint_limits_error}"
                )
            joint_rot = self.smal_3d_fitter.joint_rot
            max_limits = self.smal_3d_fitter.max_limits
            min_limits = self.smal_3d_fitter.min_limits
            zeros = torch.zeros_like(joint_rot)
            loss_limit = torch.mean(torch.max(joint_rot - max_limits, zeros) + torch.max(min_limits - joint_rot, zeros))
            loss_components["limit"] = loss_limit
            loss += self.loss_weights["w_limit"] * loss_limit

        return loss, loss_components

    def step(self, epoch):
        """Runs step of Stage, calculating loss, and running the optimiser"""

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        loss, loss_components = self.forward(new_src_mesh, iteration=epoch)

        # Optimization step
        loss.backward()
        self.optimizer.step()

        return loss, loss_components

    def plot(self):

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        figtitle = f"{self.name}, its = {self.n_it}"
        plot_meshes(
            self.target_meshes,
            new_src_mesh,
            self.mesh_names,
            title=self.name,
            figtitle=figtitle,
            out_dir=os.path.join(self.out_dir, "meshes"),
            plot_normals=self.plot_normals,
        )

    def run(self, plot=False):
        """Run the entire Stage"""

        with tqdm(np.arange(self.n_it)) as tqdm_iterator:
            for i in tqdm_iterator:
                self.optimizer.zero_grad()  # Initialise optimiser
                loss, loss_components = self.step(i)

                self.losses_to_plot.append(loss)
                if not hasattr(self, "loss_components_to_plot"):
                    self.loss_components_to_plot = {k: [] for k in loss_components.keys()}
                for k, v in loss_components.items():
                    self.loss_components_to_plot[k].append(v)

                # Print loss components at the end of each stage
                if i == self.n_it - 1:
                    print(f"\nFinal loss components for stage {self.name}:")
                    for k, v in loss_components.items():
                        print(f"{k}: {v.item():.6f}")

                tqdm_iterator.set_description(f"STAGE = {self.name}, TOT_LOSS = {loss:.6f}")

        if plot:
            self.plot()

    def save_npz(self, labels=None):
        """Given a directory, saves a .npz file of all params
        labels: optional list of size n_batch, to save as labels for all entries"""

        out = {}
        for param in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans"]:
            out[param] = getattr(self.smal_3d_fitter, param).cpu().detach().numpy()

        v = self.smal_3d_fitter()
        out["verts"] = v.cpu().detach().numpy()
        out["faces"] = self.faces.cpu().detach().numpy()
        out["labels"] = labels

        out_title = f"{self.name}.npz"
        np.savez(os.path.join(self.out_dir, out_title), **out)


class StageManager:
    """Container for multiple stages of optimisation"""

    def __init__(self, out_dir="static_fits_output", labels=None, plot_normals=False):
        """Labels: optional list of size n_batch with labels for each mesh"""
        self.stages = []
        self.out_dir = out_dir
        self.labels = labels
        self.plot_normals = plot_normals

    def run(self):
        for n, stage in enumerate(self.stages):
            stage.run(plot=config.PLOT_RESULTS)
            stage.save_npz(labels=self.labels)

        # plot loss components, total loss is plotted in the run method
        self.plot_loss_components()

    def plot_losses(self, out_src="losses"):
        """Plot combined losses for all stages."""

        fig, ax = plt.subplots()
        it_start = 0  # track number of its
        for stage in self.stages:
            out_losses = [i.cpu().detach().numpy() for i in stage.losses_to_plot]
            n_it = stage.n_it
            ax.semilogy(np.arange(it_start, it_start + n_it), out_losses, label=stage.name)
            it_start += n_it

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total loss")
        ax.legend()
        out_src = os.path.join(self.out_dir, out_src + ".png")
        plt.tight_layout()
        fig.savefig(out_src)
        plt.close(fig)

    def plot_loss_components(self, out_src="loss_components"):
        """Plot individual loss components for all stages."""

        # Get all unique loss component names across all stages
        all_components = set()
        for stage in self.stages:
            if hasattr(stage, "loss_components_to_plot"):
                all_components.update(stage.loss_components_to_plot.keys())

        # Create a subplot for each loss component
        n_components = len(all_components)
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 4 * n_components))
        if n_components == 1:
            axes = [axes]

        it_start = 0
        for stage in self.stages:
            if hasattr(stage, "loss_components_to_plot"):
                for i, component in enumerate(all_components):
                    if component in stage.loss_components_to_plot:
                        values = [v.cpu().detach().numpy() for v in stage.loss_components_to_plot[component]]
                        axes[i].semilogy(np.arange(it_start, it_start + len(values)), values, label=f"{stage.name}")
                        axes[i].set_title(f"{component} loss")
                        axes[i].set_xlabel("Epoch")
                        axes[i].set_ylabel("Loss value")
                        axes[i].legend()
            it_start += stage.n_it

        plt.tight_layout()
        out_src = os.path.join(self.out_dir, out_src + ".png")
        fig.savefig(out_src)
        plt.close(fig)

    def add_stage(self, stage):
        self.stages.append(stage)
