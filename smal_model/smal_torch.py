"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from .smal_basics import align_smal_template_to_symmetry_axis  # , get_smal_template
import torch.nn as nn
import config


# There are chumpy variables so convert them to numpy.
# correction -> there should no longer be chumpy variables and if they crop up, we should fix them
def undo_chumpy(x):
    if hasattr(x, 'r') and not isinstance(x, np.ndarray):
        print("WARNING: chumpy variable: ", x)
    try:
        return x if isinstance(x, np.ndarray) else x.r
    except AttributeError:
        return x


class SMAL(nn.Module):
    def __init__(self, device, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        # -- Load SMPL params --
        # with open(pkl_path, 'r') as f:
        #     dd = pkl.load(f)

        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()

            if config.DEBUG:
                print(config.SMAL_FILE)
                for key, value in dd.items():
                    print(key)
                    try:
                        print(value.shape)
                    except:
                        pass
                    print(value)

        self.f = dd['f']

        self.faces = torch.from_numpy(self.f.astype(int)).to(device)

        # replaced logic in here (which requried SMPL library with L58-L68)
        # v_template = get_smal_template(
        #     model_name=config.SMAL_FILE, 
        #     data_name=config.SMAL_DATA_FILE, 
        #     shape_family_id=shape_family_id)

        v_template = dd['v_template']

        # Size of mesh [Number of vertices, 3]
        self.size = [v_template.shape[0], 3]

        """
        READ IN LEARNED BLEND SHAPES
        """
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis -> betas are blend shapes(?)
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T.copy()
        self.shapedirs = Variable(
            torch.Tensor(shapedir), requires_grad=False).to(device)

        if config.DEBUG:
            print("\nBETAS AND SHAPES:")
            print(self.num_betas)
            print(self.shapedirs.shape)
            print(self.shapedirs[0][:21])

        if shape_family_id != -1:
            with open(config.SMAL_DATA_FILE, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()

            betas = data['cluster_means'][shape_family_id]
            # TODO - THESE CLUSTER MEANS ARE NOT GOING TO BE USED IN OUR SMIL MODEL FOR NOW!
            v_template = v_template + np.matmul(betas[None, :], shapedir).reshape(
                -1, self.size[0], self.size[1])[0]

        try:
            symmetry_axis_vertices = dd["sym_verts"]
        except KeyError:
            print("No symmetry axis vertices provided - using default values!")
            symmetry_axis_vertices = None

        if config.ignore_hardcoded_body:
            v_sym, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(
                v_template,
                sym_file=None,
                I=symmetry_axis_vertices)
            # symmetry file
        else:
            v_sym, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(
                v_template,
                sym_file=config.SMAL_SYM_FILE,
                I=symmetry_axis_vertices)
            # symmetry file

        # Mean template vertices
        self.v_template = Variable(
            torch.Tensor(v_sym),
            requires_grad=False).to(device)

        # Regressor for joint locations given shape
        try:
            self.J_regressor = Variable(
                torch.Tensor(dd['J_regressor'].T.todense()),
                requires_grad=False).to(device)
        except:
            # in custom Blender exporter the J_regressor is stored in dense matrix form
            self.J_regressor = Variable(
                torch.Tensor(dd['J_regressor'].T),
                requires_grad=False).to(device)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]

        # If there are no pose blend shapes, create a zeros tensor of appropriate size
        if dd['posedirs'].size != 0: # != np.empty(0):
            posedirs = np.reshape(
                undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T

            self.posedirs = Variable(
                torch.Tensor(posedirs), requires_grad=False).to(device)
        else:
            # shape joints - 1 (root bone) * 3 * 3 , vertices * 3
            posedirs = np.zeros(((self.J_regressor.shape[1] - 1) * 3 * 3,
                                 self.v_template.shape[0] * 3))

            self.posedirs = Variable(
                torch.Tensor(posedirs), requires_grad=False).to(device)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])),
            requires_grad=False).to(device)

    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True, v_template=None):

        nBetas = beta.shape[1]

        # DEBUG: set nBetas to zero, assuming no blend shapes have been registered yet.
        # comment out line, once blend shapes are included
        # nBetas = 0

        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        if v_template is None:
            v_template = self.v_template

        # 1. Add shape blend shapes

        if nBetas > 0:
            if del_v is None:

                if config.DEBUG:
                    print("size 0 : ", self.size[0])
                    print("size 1 : ", self.size[1])
                    print("beta   : ", beta.shape)
                    print("shape  : ", self.shapedirs[:nBetas, :].shape)
                    print("v_temp : ", v_template.shape)

                # repeat shapedir, in case only one shape is provided to begin with
                if self.shapedirs[:nBetas, :].shape[1] != 3 * v_template.shape[0]:
                    temp_shape = torch.reshape(self.shapedirs[:nBetas, :], [1, 3 * v_template.shape[0]])
                    temp_shape_rep = temp_shape.repeat(20, 1)
                    v_shaped = v_template + torch.reshape(torch.matmul(beta, temp_shape_rep),
                                                          [-1, self.size[0], self.size[1]])
                else:
                    v_shaped = v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :]),
                                                          [-1, self.size[0], self.size[1]])
            else:
                v_shaped = v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :]),
                                                              [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3

        # reformat pose library if needed/home/fabi/SMAL/SMALify/fit3d_results_ALL_ANTS_ALL_METHODS/Stage3.npz

        if self.posedirs.shape[1] != 3 * v_template.shape[0]:
            self.posedirs = torch.reshape(self.posedirs, [1, 3 * v_template.shape[0]])

        # get number of joints
        NUM_JOINTS = self.J_regressor.shape[1]

        if config.DEBUG:
            print("NUM_JOINTS : ", NUM_JOINTS)
            print("posedirs   : ", self.posedirs.shape)

        if theta.shape[1] != NUM_JOINTS:
            if nBetas > 0:
                dim_x = beta.shape[0]
            else:
                dim_x = 1
            theta = torch.zeros(dim_x, NUM_JOINTS, 3).to(beta.device)

        if len(theta.shape) == 4:
            Rs = theta
        else:
            Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, NUM_JOINTS, 3, 3])

        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, (NUM_JOINTS - 1) * 3 * 3])

        """
        if pose_feature.shape[0] > pose_feature.shape[1]:
            pose_feature = torch.zeros([1, 1]).to(beta.device)
        """

        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        # 4. Get the global joint location
        # DEBUG - delete once betas are provided
        # betas_logscale = None

        self.J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents, betas_logscale=betas_logscale,
            num_joints=NUM_JOINTS)

        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, NUM_JOINTS])

        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, NUM_JOINTS, 16])),
            [num_batch, -1, 4, 4])

        if config.DEBUG:
            print("\nv_posed    : ", v_posed.shape)
            print("Rs           : ", Rs.shape)
            print("num_batch    : ", num_batch)
            print("pose_feature : ", pose_feature.shape)
            print("T : ", T.shape)

        v_posed_homo = torch.cat(
            [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch, 3)).to(device=beta.device)

        verts = verts + trans[:, None, :]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if NUM_JOINTS == 35:  # assuming configuration of WLDO and SMAL is used:
            joints = torch.cat([
                joints,
                verts[:, None, 1863],  # end_of_nose
                verts[:, None, 26],  # chin
                verts[:, None, 2124],  # right ear tip
                verts[:, None, 150],  # left ear tip
                verts[:, None, 3055],  # left eye
                verts[:, None, 1097],  # right eye
            ], dim=1)

        if get_skin:
            return verts, joints, Rs, v_shaped
        else:
            return joints
