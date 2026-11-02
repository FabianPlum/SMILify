"""
Tests for the transformer decoder head's IEF iteration history.
"""

import math

import torch

from smal_fitter.neuralSMIL.transformer_decoder import SMILTransformerDecoderHead


def test_iteration_history_mesh_scale_is_log_space():
    """The IEF history holds log mesh scales; only the final output is exponentiated.

    The multi-view IEF intermediate keypoint loss exponentiates the history entries, so it relies on this.
    """
    torch.manual_seed(0)
    head = SMILTransformerDecoderHead(
        feature_dim=16,
        context_dim=16,
        hidden_dim=32,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        ief_iters=3,
        rotation_representation="6d",
        scale_trans_mode="ignore",
        allow_mesh_scaling=True,
        mesh_scale_init=2.0,
    ).eval()
    with torch.no_grad():
        out = head(torch.randn(4, 16), torch.randn(4, 5, 16))

    history = out["iteration_history"]["mesh_scale"]
    assert len(history) == 3
    torch.testing.assert_close(out["mesh_scale"], torch.exp(history[-1]))
    # mesh_scale_init=2.0 starts the IEF at log(2.0) ~ 0.69; a linear history would start near 2.0
    assert torch.allclose(history[0], torch.full_like(history[0], math.log(2.0)), atol=0.1)
