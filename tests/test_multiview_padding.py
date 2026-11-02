"""Padded multi-view batches: padded view slots skip the backbone and do not change a sample's prediction."""

import pytest
import torch
import torch.nn as nn

from smal_fitter.neuralSMIL.multiview_smil_regressor import (
    CameraHead,
    MultiViewFeatureFusion,
    MultiViewSMILImageRegressor,
)
from smal_fitter.neuralSMIL.transformer_decoder import SMILTransformerDecoderHead

DIM = 16


class _CountingBackbone(nn.Module):
    """ViT-like stand-in: an 8x8 image becomes 4 patch tokens and a pooled global feature. Counts its images."""

    def __init__(self):
        super().__init__()
        self.patchify = nn.Conv2d(3, DIM, kernel_size=4, stride=4)
        self.pool = nn.Linear(DIM, DIM)
        self.images_seen = 0

    def forward_with_spatial(self, x):
        self.images_seen += x.shape[0]
        patches = self.patchify(x).flatten(2).transpose(1, 2)  # (N, 4, DIM)
        return self.pool(patches.mean(dim=1)), patches

    def forward(self, x):
        return self.forward_with_spatial(x)[0]

    def get_spatial_dim(self):
        return DIM


def _tiny_regressor(backbone_chunk_size):
    """The real forward_multiview on scaled-down modules: transformer_decoder head, 3 cameras, no SMAL model."""
    torch.manual_seed(0)
    model = object.__new__(MultiViewSMILImageRegressor)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.head_type = "transformer_decoder"
    model.feature_dim = DIM
    model.scale_trans_mode = "ignore"
    model.num_canonical_cameras = 3
    model.use_gt_camera_init = False
    model.backbone_chunk_size = backbone_chunk_size
    model.backbone = _CountingBackbone()
    model.view_fusion = MultiViewFeatureFusion(DIM, num_layers=1, num_heads=2, dropout=0.0)
    model.view_embeddings = nn.Embedding(3, DIM)
    model.patch_view_embed = nn.Embedding(3, DIM)
    model.camera_heads = nn.ModuleList([CameraHead(DIM, hidden_dim=16) for _ in range(3)])
    model.transformer_head = SMILTransformerDecoderHead(
        feature_dim=DIM,
        context_dim=DIM,
        hidden_dim=32,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        ief_iters=2,
        rotation_representation="6d",
        scale_trans_mode="ignore",
    )
    with torch.no_grad():
        for p in model.parameters():  # the heads' near-zero init would hide what the decoder attends to
            p.normal_(0.0, 0.5)
    return model.eval()


@pytest.mark.parametrize("backbone_chunk_size", [None, 3])
def test_padded_sample_matches_the_sample_alone(backbone_chunk_size):
    model = _tiny_regressor(backbone_chunk_size)
    torch.manual_seed(1)
    views_a = torch.rand(3, 3, 8, 8)  # sample a: 3 views (cameras 0, 1, 2)
    view_b = torch.rand(3, 8, 8)  # sample b: 1 view (camera 2), padded to 3 in the batch
    pad = torch.zeros(3, 8, 8)
    images_per_view = [
        torch.stack([views_a[0], view_b]),
        torch.stack([views_a[1], pad]),
        torch.stack([views_a[2], pad]),
    ]
    camera_indices = torch.tensor([[0, 1, 2], [2, 0, 0]])
    view_mask = torch.tensor([[True, True, True], [True, False, False]])

    with torch.no_grad():
        batch = model.forward_multiview(images_per_view, camera_indices, view_mask)
        assert model.backbone.images_seen == 4  # the 2 padded slots never reach the backbone
        alone_a = model.forward_multiview([v[None] for v in views_a], camera_indices[:1], view_mask[:1])
        alone_b = model.forward_multiview([view_b[None]], camera_indices[1:, :1], view_mask[1:, :1])

    # float32 noise from the different batch shapes is about 2e-5 here; before the decoder masked
    # padded views, the padded sample's body parameters were off by about 15
    close = dict(rtol=1e-3, atol=1e-3)
    for key in ("global_rot", "joint_rot", "betas", "trans"):
        torch.testing.assert_close(batch[key][0], alone_a[key][0], **close)
        torch.testing.assert_close(batch[key][1], alone_b[key][0], **close)
    for key in ("fov_per_view", "cam_rot_per_view", "cam_trans_per_view"):
        for v in range(3):
            torch.testing.assert_close(batch[key][v][0], alone_a[key][v][0], **close)
        torch.testing.assert_close(batch[key][0][1], alone_b[key][0][0], **close)
