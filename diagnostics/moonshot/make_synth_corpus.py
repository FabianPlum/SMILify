"""Generate synthetic targets with KNOWN per-vertex correspondence, optionally noisy.

WHY THIS EXISTS
Every correspondence number in this report is indirect. probe-19 measures whether the fit's
vertex placement is CONSISTENT across a population, never whether it is CORRECT -- a
systematically wrong but reproducible correspondence scores perfectly. Its synthetic control
validated the METRIC (shapes drawn straight from the shape space) and never the PIPELINE.

Here the target is generated FROM the model, so ground-truth correspondence is known exactly:
fitted vertex i should land on generated vertex i. No population inference, no proxy.

This is a CEILING test. The targets are the model's own geometry, so if the pipeline cannot
establish correspondence here it certainly cannot on ethanol-preserved worker scans. Noise is
added on top to stop it being trivially easy and to bracket real scan roughness (§6 measured
worker median dihedral 10.54 deg and edge-length CV 0.43 against the reference's 8.20 / 0.30).

Usage:
  python diagnostics/moonshot/make_synth_corpus.py --n 12 --noise 0.0 --out synth_clean
  python diagnostics/moonshot/make_synth_corpus.py --n 12 --noise 0.005 --out synth_noisy
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from fitter_3d.pointcloud2smil.sample_smil_model import (  # noqa: E402
    export_mesh_to_obj,
    generate_random_parameters,
)
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--noise", type=float, default=0.0, help="per-vertex gaussian sd, in model units")
    ap.add_argument("--pose_scale", type=float, default=0.25)
    ap.add_argument("--shape_scale", type=float, default=1.0)
    ap.add_argument("--scale_scale", type=float, default=0.10, help="per-joint log-scale sd")
    ap.add_argument("--out", default="synth")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(HERE, args.out)
    os.makedirs(outdir, exist_ok=True)

    fitter = SMAL3DFitter(batch_size=args.n, device=dev, shape_family=-1)
    generate_random_parameters(
        fitter,
        seed=args.seed,
        random_dist="normal",
        shape_scale=args.shape_scale,
        pose_scale=args.pose_scale,
        trans_scale=0.0,  # translation is removed by load_meshes' centring anyway
        scale_scale=args.scale_scale,
        global_rot_scale=0.0,  # scans are canonically aligned (§4), so keep the synthetic set so too
    )
    with torch.no_grad():
        verts = fitter()  # (n, V, 3) -- GROUND TRUTH, index i is anatomical point i
    faces = fitter.faces

    rng = np.random.default_rng(args.seed)
    gt, names = [], []
    for i in range(args.n):
        v = verts[i]
        if args.noise > 0:
            v = v + torch.tensor(rng.normal(0.0, args.noise, v.shape), dtype=v.dtype, device=v.device)
        name = f"synth_{i:03d}"
        export_mesh_to_obj(v, faces, os.path.join(outdir, f"{name}.obj"))
        # store the CLEAN ground truth: correspondence is defined by the noiseless geometry,
        # and scoring against the noisy copy would credit the fit for reproducing the noise
        gt.append(verts[i].cpu().numpy())
        names.append(name)

    ext = float((verts.max(1).values - verts.min(1).values).max())
    np.savez(
        os.path.join(outdir, "ground_truth.npz"),
        verts=np.stack(gt),
        names=np.array(names),
        betas=fitter.betas.detach().cpu().numpy(),
        joint_rot=fitter.joint_rot.detach().cpu().numpy(),
        log_beta_scales=fitter.log_beta_scales.detach().cpu().numpy(),
        noise=args.noise,
        extent=ext,
    )
    print(
        f"\n[synth] {args.n} targets in {outdir}, noise sd {args.noise} "
        f"({100 * args.noise / ext:.2f}% of extent {ext:.3f})"
    )
    print(f"[synth] ground truth -> {os.path.join(outdir, 'ground_truth.npz')}")


if __name__ == "__main__":
    main()
