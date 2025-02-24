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
from fitter_3d.utils import plot_pointclouds, plot_meshes
import numpy as np
import os
import config
import pickle as pkl

from smal_model.smal_torch import SMAL
from smal_fitter.utils import eul_to_axis

nn = torch.nn

default_weights = dict(w_chamfer=1.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1)

# Want to vary learning ratios between parameters,
default_lr_ratios = []


def get_meshes(verts, faces, device='cuda'):
    """Returns Meshes object of all SMAL meshes."""
    meshes = Meshes(verts=verts, faces=faces).to(device)
    return meshes


class SMAL3DFitter(nn.Module):
    def __init__(self, batch_size=1, device='cuda', shape_family=-1):
        super(SMAL3DFitter, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_betas = config.N_BETAS


        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()
        
        if config.ignore_hardcoded_body:
            model_covs = dd['shape_cov']
            self.mean_betas = torch.FloatTensor(dd['shape_mean_betas']).to(device)

        else:
            with open(config.SMAL_DATA_FILE, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                smal_data = u.load()

                self.shape_family_list = np.array(shape_family)

                model_covs = np.array(smal_data['cluster_cov'])[[shape_family]][0]
                self.mean_betas = torch.FloatTensor(smal_data['cluster_means'][[shape_family]][0])[:config.N_BETAS].to(device)

        if config.DEBUG:
            print("MODEL COVS", model_covs)

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[:config.N_BETAS, :config.N_BETAS].to(device)

        if config.DEBUG:
            print("MEAN BETAS", self.mean_betas)

        self.betas = nn.Parameter(
            self.mean_betas.unsqueeze(0).repeat(batch_size, 1))

        # Load the kinematic tree from SMAL model data
        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()
            self.kintree_table = torch.tensor(dd['kintree_table']).to(device)
        
        # Get number of joints from kintree
        self.n_joints = self.kintree_table.shape[1]
        
        # Initialize log_beta_scales with proper shape: batch_size x n_joints x 3
        # Starting with ones means no scaling initially (since exp(0) = 1)
        self.log_beta_scales = nn.Parameter(
            torch.zeros(self.batch_size, self.n_joints, 3).to(device))

        global_rotation_np = eul_to_axis(np.array([0, 0, 0]))
        global_rotation = torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(batch_size,
                                                                                                      1)  # Global Init (Head-On)
        self.global_rot = nn.Parameter(global_rotation)

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(batch_size, 1)  # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(batch_size, config.N_POSE, 3).to(device)
        self.joint_rot = nn.Parameter(default_joints)

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0

        # Can be used to prevent certain joints rotating.
        # Can be useful depending on sequence.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device) # by default all joints are free to rotate
        # self.rotation_mask[25:32] = 0.0 # e.g. stop the tail moving

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(device, shape_family_id=shape_family)
        self.faces = self.smal_model.faces.unsqueeze(0).repeat(batch_size, 1, 1)

        """
        # vertex offsets for deformations
        self.deform_verts = nn.Parameter(torch.zeros(batch_size, *self.smal_model.v_template.shape)).to(device)
        """

        # Initialize deform_verts as a leaf tensor
        self.deform_verts = nn.Parameter(torch.zeros(batch_size, *self.smal_model.v_template.shape).to(device),
                                         requires_grad=True)

    def get_joint_scales(self):
        """
        Compute the final scale for each joint taking into account the kinematic chain.
        A joint's scale is influenced by its own scale parameters and all its parents.
        """
        # Start with base scales from log_beta_scales
        joint_scales = torch.exp(self.log_beta_scales)  # Convert from log space
        
        # For each joint
        for joint_idx in range(self.n_joints):
            parent_idx = self.kintree_table[0, joint_idx]
            
            # If this joint has a parent (parent_idx != joint_idx)
            if parent_idx != joint_idx:
                # Accumulate parent's scale
                joint_scales[:, joint_idx] *= joint_scales[:, parent_idx]
        
        return joint_scales

    def forward(self):
        # Get accumulated joint scales
        joint_scales = self.get_joint_scales()
        
        # Reshape to match expected format: batch_size x num_joints x 3
        betas_logscale = self.log_beta_scales.reshape(self.batch_size, -1, 3)
        
        verts, joints, Rs, v_shaped = self.smal_model(
            self.betas,
            torch.cat([
                self.global_rot.unsqueeze(1),
                self.joint_rot], dim=1),
            betas_logscale=betas_logscale)  # Pass properly shaped tensor

        verts = verts + self.trans.unsqueeze(1)
        joints = joints + self.trans.unsqueeze(1)

        verts += self.deform_verts

        return verts


class SMALParamGroup:
    """Object building on model.parameters, with modifications such as variable learning rate"""
    param_map = {
        "init": ["global_rot", "trans"],
        "init_rot_lock": ["trans"],
        "default": ["global_rot", "joint_rot", "trans", "betas", "log_beta_scales"],
        "shape": ["global_rot", "trans", "betas", "log_beta_scales"],
        "pose": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales"],
        "deform": ["deform_verts"],
        "all": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "deform_verts"]
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

    def __init__(self, nits: int, scheme: str, smal_3d_fitter: SMAL3DFitter, target_meshes: Meshes, mesh_names=[],
                 name="optimise",
                 loss_weights=None, lr=1e-3, out_dir="static_fits_output",
                 custom_lrs=None, device='cuda'):
        """
        nits = integer, number of iterations in stage
        parameters = list of items over which to be optimised
        get_mesh = function that returns Mesh object for identifying losses
        name = name of stage

        lr_decay = factor by which lr decreases at each it"""

        self.n_it = nits
        self.name = name
        self.out_dir = out_dir
        self.target_meshes = target_meshes
        self.mesh_names = mesh_names
        self.smal_3d_fitter = smal_3d_fitter
        self.device = device

        self.loss_weights = default_weights.copy()
        if loss_weights is not None:
            for k, v in loss_weights.items():
                self.loss_weights[k] = v

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

        self.consider_loss = lambda loss_name: self.loss_weights[
                                                   f"w_{loss_name}"] > 0  # function to check if loss is non-zero

    def forward(self, src_mesh):
        loss = 0

        # Sample from target meshes
        target_verts = sample_points_from_meshes(self.target_meshes, 3000)

        if self.consider_loss("chamfer"):
            loss_chamfer, _ = chamfer_distance(target_verts, src_mesh.verts_padded())
            loss += self.loss_weights["w_chamfer"] * loss_chamfer

        if self.consider_loss("edge"):
            loss_edge = mesh_edge_loss(src_mesh)  # and (b) the edge length of the predicted mesh
            loss += self.loss_weights["w_edge"] * loss_edge

        if self.consider_loss("normal"):
            loss_normal = mesh_normal_consistency(src_mesh)  # mesh normal consistency
            loss += self.loss_weights["w_normal"] * loss_normal

        if self.consider_loss("laplacian"):
            loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")  # mesh laplacian smoothing
            loss += self.loss_weights["w_laplacian"] * loss_laplacian

        return loss

    def step(self, epoch):
        """Runs step of Stage, calculating loss, and running the optimiser"""

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        loss = self.forward(new_src_mesh)

        # Optimization step
        loss.backward()
        self.optimizer.step()

        return loss

    def plot(self):

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        figtitle = f"{self.name}, its = {self.n_it}"
        plot_meshes(self.target_meshes, new_src_mesh, self.mesh_names, title=self.name,
                    figtitle=figtitle,
                    out_dir=os.path.join(self.out_dir, "meshes"))

    def run(self, plot=False):
        """Run the entire Stage"""

        with tqdm(np.arange(self.n_it)) as tqdm_iterator:
            for i in tqdm_iterator:
                self.optimizer.zero_grad()  # Initialise optimiser
                loss = self.step(i)

                self.losses_to_plot.append(loss)

                tqdm_iterator.set_description(
                    f"STAGE = {self.name}, TOT_LOSS = {loss:.6f}")  # Print the losses

        if plot:
            self.plot()

    def save_npz(self, labels=None):
        """Given a directory, saves a .npz file of all params
        labels: optional list of size n_batch, to save as labels for all entries"""

        out = {}
        for param in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts"]:
            out[param] = getattr(self.smal_3d_fitter, param).cpu().detach().numpy()

        v = self.smal_3d_fitter()
        out["verts"] = v.cpu().detach().numpy()
        out["faces"] = self.faces.cpu().detach().numpy()
        out["labels"] = labels

        out_title = f"{self.name}.npz"
        np.savez(os.path.join(self.out_dir, out_title), **out)


class StageManager:
    """Container for multiple stages of optimisation"""

    def __init__(self, out_dir="static_fits_output", labels=None):
        """Labels: optional list of size n_batch with labels for each mesh"""
        self.stages = []
        self.out_dir = out_dir
        self.labels = labels

    def run(self):
        for n, stage in enumerate(self.stages):
            stage.run(plot=config.PLOT_RESULTS)
            stage.save_npz(labels=self.labels)

        # commented out for now, as we plot losses in the run method
        #self.plot_losses()

    def plot_losses(self, out_src="losses"):
        """Plot combined losses for all stages."""

        fig, ax = plt.subplots()
        it_start = 0  # track number of its
        for stage in self.stages:
            out_losses = [i.cpu().detach().numpy() for i in stage.losses_to_plot]
            n_it = stage.n_it
            ax.semilogy(np.arange(it_start, it_start + n_it), out_losses, label=stage.name)
            it_start += n_it

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total loss')
        ax.legend()
        out_src = os.path.join(self.out_dir, out_src + ".png")
        plt.tight_layout()
        fig.savefig(out_src)
        plt.close(fig)

    def add_stage(self, stage):
        self.stages.append(stage)
