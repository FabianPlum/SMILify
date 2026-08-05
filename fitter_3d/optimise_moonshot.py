"""Runner for the moonshot registration experiments.

Isolated from fitter_3d/optimise.py; the live path is untouched. Adds:
  * per-stage robust-kernel / trimming / sampling controls
  * an evaluation pass that writes a per-specimen metrics CSV using
    diagnostics/moonshot/metrics.py, so every experiment is directly comparable
  * deterministic seeding, so seed-repeat experiments are meaningful

Usage:
  python -m fitter_3d.optimise_moonshot --mesh_dir <dir> --yaml_src <cfg.yaml> [--seed 0]
"""

import argparse
import gc
import glob
import os
import sys
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

if os.getcwd().endswith("fitter_3d"):
    os.chdir("../")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagnostics", "moonshot")))

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage, MoonshotManager  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir", type=str, required=True)
    p.add_argument("--yaml_src", type=str, required=True)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape_family_id", type=int, default=-1)
    p.add_argument("--max_meshes", type=int, default=-1)
    p.add_argument("--eval", action="store_true", help="run the metric suite after fitting")
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="path to a stage .npz whose pose/shape state initialises this run. "
        "Used to hand a better pose (e.g. from the hierarchical fitter) to a "
        "standard surface-convergence schedule, so the two contributions are "
        "separable.",
    )
    return p


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.yaml_src) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    stage_options = cfg["stages"]
    for k, v in (cfg.get("args") or {}).items():
        if getattr(args, k, None) is None or k not in ("results_dir",):
            setattr(args, k, v)
    if args.results_dir is None:
        args.results_dir = cfg.get("args", {}).get("results_dir", "moonshot_results")

    os.makedirs(args.results_dir, exist_ok=True)

    mesh_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    if args.max_meshes > 0:
        mesh_files = mesh_files[: args.max_meshes]
    mesh_names = [os.path.basename(f) for f in mesh_files]
    print(f"[moonshot] {len(mesh_files)} meshes from {args.mesh_dir}", flush=True)

    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    n = len(target_meshes)

    smal = SMAL3DFitter(batch_size=n, device=device, shape_family=args.shape_family_id)

    if args.init_from:
        d = np.load(args.init_from, allow_pickle=True)
        with torch.no_grad():
            for k in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans", "deform_verts"]:
                if k in d:
                    getattr(smal, k).data = torch.tensor(d[k], dtype=torch.float32, device=device)
        print(f"[moonshot] initialised pose/shape from {args.init_from}", flush=True)

    manager = MoonshotManager(out_dir=args.results_dir, labels=mesh_names)

    common = dict(
        target_meshes=target_meshes,
        smal_3d_fitter=smal,
        out_dir=args.results_dir,
        device=device,
        mesh_names=mesh_names,
    )

    for stage_name, kw in stage_options.items():
        kw = dict(kw)
        stage = MoonshotStage(name=stage_name, **kw, **common)
        manager.add_stage(stage)

    print(f"[moonshot] running {len(manager.stages)} stages", flush=True)
    manager.run()
    manager.plot_losses("losses")

    if args.eval:
        run_eval(args, smal, target_meshes, mesh_names, device)

    del smal, manager, target_meshes
    torch.cuda.empty_cache()
    gc.collect()


def run_eval(args, smal, target_meshes, mesh_names, device):
    """Evaluate the final fit with the full metric suite and write a CSV."""
    import csv
    import pickle

    import metrics as M

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    sym_verts = torch.tensor(np.asarray(dd["sym_verts"]).astype(np.int64))
    weights = np.asarray(dd["weights"])
    jnames = list(dd["J_names"])
    part_ids, part_names = M.template_part_segmentation(weights, jnames)

    with torch.no_grad():
        pred = smal().detach()
        rest = smal(deform_verts=torch.zeros_like(smal.deform_verts)).detach()
        dv = smal.deform_verts.detach()
    faces = smal.faces[0].detach()

    rows = []
    for i, name in enumerate(mesh_names):
        tgt = target_meshes[i]
        r = M.evaluate(
            pred[i],
            faces,
            tgt,
            rest[i],
            sym_verts,
            part_ids,
            part_names,
            deform_verts=dv[i],
            device=device,
        )
        r["mesh"] = name
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  eval {i + 1}/{len(mesh_names)}", flush=True)

    keys = ["mesh"] + [k for k in rows[0] if k != "mesh"]
    out_csv = os.path.join(args.results_dir, "metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[moonshot] wrote {out_csv}", flush=True)

    # concise console summary of the headline metrics
    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n  ---- summary ----")
    for k in [
        "chamfer_l2",
        "fscore@0.01",
        "fscore@0.02",
        "hausdorff_95",
        "normal_consistency",
        "edge_logratio_absmean",
        "tri_quality_mean",
        "deform_mag_mean",
        "midline_dev_mean",
        "part_leg_distal_dist_mean",
        "part_leg_dist_mean",
        "part_body_dist_mean",
    ]:
        if k in rows[0]:
            print(f"    {k:<32} {mean(k):.6f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
