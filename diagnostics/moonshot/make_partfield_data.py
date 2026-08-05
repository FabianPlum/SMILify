"""Build the part-field training corpus. No manual annotation anywhere in this file.

TWO LABEL SOURCES, deliberately kept separate so their contributions are separable:

  SYNTHETIC (exact labels).  Pose/shape parameters are resampled from the fits this project
  already produced on the bench specimens, interpolated between pairs and jittered, then
  pushed back through the SMAL model. Labels come straight from `weights.argmax` and are
  exact -- there is no correspondence estimate anywhere in the chain. On top of that go
  augmentations chosen to match what CT scans of ethanol-preserved ants actually look like:
  surface noise, holes, non-uniform density, and debris (blobs, and "bridges" of medium
  spanning two surface points, which is what makes real legs appear fused).

  REAL-WEAK (noisy labels).  Target points from the bench scans, labelled by nearest
  neighbour on the corresponding M7 fitted mesh. In-domain but inheriting every error the
  fit made -- which is exactly the circularity this whole exercise exists to escape. They
  are written to a separate file so training can be run with and without them and the
  difference measured, rather than assumed.

SPLIT.  Bench specimens are split 35/15. Synthetic poses are drawn ONLY from train-split
fits and real-weak labels ONLY from train-split scans, so the 15 held-out specimens are
untouched by any label. The unsupervised gates (G1, G2, G3) run on all 50 regardless, since
they need no labels at all.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.partfield import part_names, template_vertex_labels, normalise  # noqa: E402

OUT = os.path.join(HERE, "partfield")
N_PTS = 4096
DEBRIS_THRESH = 0.035  # normalised units; scan points further than this from the fit


# ------------------------------------------------------------------ sampling with labels
def sample_with_labels(verts, faces, vlabel, n, gen):
    """Area-weighted surface sampling that carries a per-vertex integer label.

    The label of a sampled point is that of the barycentrically DOMINANT vertex, not an
    interpolation -- labels are categorical. Points near a part boundary therefore get the
    label of whichever part owns most of their triangle, which is the correct behaviour for
    a partition (boundaries are a measure-zero problem; interior purity is what matters).
    """
    v = verts[faces]  # (F,3,3)
    area = 0.5 * torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1).norm(dim=-1)
    area = area.clamp_min(0)
    if float(area.sum()) <= 0:
        raise ValueError("degenerate mesh: zero total area")
    fi = torch.multinomial(area, n, replacement=True, generator=gen)
    u = torch.rand(n, 1, device=verts.device, generator=gen)
    w2 = torch.rand(n, 1, device=verts.device, generator=gen)
    flip = (u + w2) > 1
    u = torch.where(flip, 1 - u, u)
    w2 = torch.where(flip, 1 - w2, w2)
    bc = torch.cat([1 - u - w2, u, w2], dim=1)  # (n,3)
    pts = (bc.unsqueeze(-1) * v[fi]).sum(1)
    lab = vlabel[faces[fi]].gather(1, bc.argmax(1, keepdim=True)).squeeze(1)
    return pts, lab


# ------------------------------------------------------------------ CT-like augmentations
def augment(pts, lab, debris_id, gen, rng):
    """Make a clean model sample look like a CT scan of an ant in ethanol.

    Each augmentation exists because of a specific property of this data, noted inline.
    Operates on NORMALISED coordinates so the magnitudes are interpretable as fractions of
    the specimen half-extent.
    """
    dev = pts.device
    n = pts.shape[0]

    # (1) surface noise -- reconstruction jitter on a marching-cubes isosurface
    sig = float(rng.uniform(0.0008, 0.006))
    pts = pts + torch.randn(pts.shape, device=dev, generator=gen) * sig

    # (2) holes -- thin structures drop out of the isosurface where contrast is poor. The
    #     tarsi in this dataset are frequently partly missing, so the network must not
    #     require a complete leg to label the part of it that survived.
    keep = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(int(rng.integers(0, 4))):
        c = pts[int(rng.integers(0, n))]
        r = float(rng.uniform(0.03, 0.11))
        keep &= (pts - c).norm(dim=-1) > r

    # (3) debris -- mounting medium and detached fragments, ~always present in these scans
    extra_p, extra_l = [], []
    for _ in range(int(rng.integers(0, 3))):
        c = pts[int(rng.integers(0, n))] + torch.randn(3, device=dev, generator=gen) * 0.05
        k = int(rng.integers(40, 260))
        extra_p.append(c + torch.randn(k, 3, device=dev, generator=gen) * float(rng.uniform(0.01, 0.05)))
        extra_l.append(torch.full((k,), debris_id, dtype=torch.long, device=dev))

    # (4) debris BRIDGES -- a filament of medium spanning two surface points. This is the
    #     augmentation that matters most: it is what makes two legs appear fused, and it is
    #     precisely the configuration where the geodesic level-set method collapsed (probe
    #     14). A network that has seen bridges can learn to label across one.
    for _ in range(int(rng.integers(0, 3))):
        a, b = pts[int(rng.integers(0, n))], pts[int(rng.integers(0, n))]
        if float((a - b).norm()) > 0.7:
            continue
        k = int(rng.integers(30, 150))
        t = torch.rand(k, 1, device=dev, generator=gen)
        seg = a * (1 - t) + b * t + torch.randn(k, 3, device=dev, generator=gen) * 0.008
        extra_p.append(seg)
        extra_l.append(torch.full((k,), debris_id, dtype=torch.long, device=dev))

    pts, lab = pts[keep], lab[keep]
    if extra_p:
        pts = torch.cat([pts] + extra_p, 0)
        lab = torch.cat([lab] + extra_l, 0)

    # (5) non-uniform density -- one side of the specimen is nearer the detector
    axis = torch.randn(3, device=dev, generator=gen)
    axis = axis / axis.norm()
    proj = pts @ axis
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    p_keep = 1.0 - float(rng.uniform(0.0, 0.5)) * proj
    keep = torch.rand(pts.shape[0], device=dev, generator=gen) < p_keep
    if int(keep.sum()) > N_PTS:
        pts, lab = pts[keep], lab[keep]

    # (6) residual alignment error -- the scans are hand-canonicalised, not perfect
    ang = torch.randn(3, device=dev, generator=gen) * float(rng.uniform(0.0, 0.05))
    th = ang.norm().clamp_min(1e-9)
    k_ = (ang / th).unsqueeze(0)
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k_[0, 2], k_[0, 1], k_[0, 2], -k_[0, 0], -k_[0, 1], k_[0, 0]
    R = torch.eye(3, device=dev) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    pts = pts @ R.T

    # resample to a fixed count
    if pts.shape[0] < N_PTS:
        idx = torch.randint(0, pts.shape[0], (N_PTS,), device=dev, generator=gen)
    else:
        idx = torch.randperm(pts.shape[0], device=dev, generator=gen)[:N_PTS]
    return pts[idx], lab[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--fit_npz", default=os.path.join(HERE, "runs", "M7_handoff_midline", "Stage_3_deform_fine.npz"))
    ap.add_argument("--n_synth", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--skip_synth", action="store_true", help="reuse existing synth_*.npz and regenerate only the real-weak set"
    )
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    with open(config.SMAL_FILE, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jnames = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
    names = part_names(jnames)
    debris_id = names.index("debris")
    vlab_np = template_vertex_labels(dd["weights"], jnames)
    vlab = torch.tensor(vlab_np, device=dev)
    print(f"[data] {len(names)} classes: {names}")
    print(
        f"[data] template vertices per class: {dict(zip(names, np.bincount(vlab_np, minlength=len(names)).tolist()))}"
    )

    fit = np.load(args.fit_npz)
    n_spec = fit["betas"].shape[0]
    files = sorted(f for f in os.listdir(args.mesh_dir) if f.endswith(".obj"))
    assert len(files) == n_spec, f"{len(files)} meshes vs {n_spec} fits"

    perm = rng.permutation(n_spec)
    tr, te = np.sort(perm[:35]), np.sort(perm[35:])
    json.dump(
        dict(
            train=[files[i] for i in tr],
            test=[files[i] for i in te],
            train_idx=tr.tolist(),
            test_idx=te.tolist(),
            classes=names,
            seed=args.seed,
        ),
        open(os.path.join(OUT, "split.json"), "w"),
        indent=1,
    )
    print(f"[data] split: {len(tr)} train specimens, {len(te)} held out")

    smal = SMAL3DFitter(batch_size=1, device=dev)
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=dev)

    P = {
        k: torch.tensor(fit[k], dtype=torch.float32, device=dev)
        for k in ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans")
    }

    # ---------------------------------------------------------------- synthetic
    def gen_synth(n, pool, tag):
        pts_all = np.empty((n, N_PTS, 3), np.float32)
        lab_all = np.empty((n, N_PTS), np.int64)
        for i in range(n):
            a, b = int(rng.choice(pool)), int(rng.choice(pool))
            t = float(rng.uniform(0, 1))
            kw = {}
            for k, jit in (
                ("global_rot", 0.03),
                ("joint_rot", 0.06),
                ("betas", 0.25),
                ("log_beta_scales", 0.05),
                ("trans", 0.01),
                ("betas_trans", 0.01),
            ):
                mix = P[k][a] * (1 - t) + P[k][b] * t
                sd = P[k].std(0).clamp_min(1e-6) * jit
                kw[k] = (mix + torch.randn(mix.shape, device=dev, generator=gen) * sd).unsqueeze(0)
            with torch.no_grad():
                v = smal(**kw)
                v = v[0] if isinstance(v, (tuple, list)) else v
            p, l = sample_with_labels(v[0], faces, vlab, N_PTS * 3, gen)
            p, _, _ = normalise(p)
            p, l = augment(p, l, debris_id, gen, rng)
            p, _, _ = normalise(p)
            pts_all[i], lab_all[i] = p.cpu().numpy(), l.cpu().numpy()
            if (i + 1) % 250 == 0:
                print(f"[data] {tag} {i + 1}/{n}", flush=True)
        return pts_all, lab_all

    if args.skip_synth:
        print("[data] --skip_synth: reusing existing synth_train/synth_val")
    else:
        ptr, ltr = gen_synth(args.n_synth, tr, "synth-train")
        np.savez_compressed(os.path.join(OUT, "synth_train.npz"), pts=ptr, lab=ltr)
        pva, lva = gen_synth(args.n_val, te, "synth-val")
        np.savez_compressed(os.path.join(OUT, "synth_val.npz"), pts=pva, lab=lva)
        cnt = np.bincount(ltr.ravel(), minlength=len(names))
        print(
            "[data] synthetic class balance (train): "
            + ", ".join(f"{n}={100 * c / cnt.sum():.2f}%" for n, c in zip(names, cnt))
        )

    # ---------------------------------------------------------------- real-weak
    from pytorch3d.ops import knn_points

    _, meshes = load_meshes(mesh_dir=args.mesh_dir, sorting=sorted, device=str(dev))
    from pytorch3d.ops import sample_points_from_meshes

    fitv = torch.tensor(fit["verts"], dtype=torch.float32, device=dev)
    rp, rl, rd = [], [], []
    for i in range(n_spec):
        tgt = sample_points_from_meshes(meshes[i], 30000)[0]
        # normalise target and fit TOGETHER so the debris threshold is in shared units
        both = torch.cat([tgt, fitv[i]], 0)
        _, c, s = normalise(both)
        tn, fn = (tgt - c) / s, (fitv[i] - c) / s
        kn = knn_points(tn.unsqueeze(0), fn.unsqueeze(0), K=1)
        lab = vlab[kn.idx[0, :, 0]].clone()
        d = kn.dists[0, :, 0].sqrt()
        lab[d > DEBRIS_THRESH] = debris_id
        rd.append(float((d > DEBRIS_THRESH).float().mean()))
        rp.append(tn.cpu().numpy().astype(np.float32))
        rl.append(lab.cpu().numpy().astype(np.int64))
    # Per-specimen debris fraction IS the fit-quality signal, and it has to be kept.
    # A specimen where 46% of scan points sit further than DEBRIS_THRESH from the fitted
    # mesh does not have 46% debris -- it has a failed fit, so its weak labels are noise
    # dressed as supervision. Training filters on this rather than trusting all 50 equally.
    rd = np.array(rd, dtype=np.float32)
    np.savez_compressed(
        os.path.join(OUT, "real_weak.npz"),
        pts=np.stack(rp),
        lab=np.stack(rl),
        debris_frac=rd,
        train_idx=tr,
        test_idx=te,
        files=np.array(files),
    )
    print(
        f"[data] real-weak: debris fraction mean {100 * rd.mean():.1f}% "
        f"(min {100 * rd.min():.1f}%, max {100 * rd.max():.1f}%) at thresh {DEBRIS_THRESH}"
    )
    bad = np.argsort(-rd)[:5]
    print("[data] worst-fit specimens (weak labels here are unreliable):")
    for i in bad:
        print(f"        {100 * rd[i]:5.1f}%  {files[i]}")
    print(f"[data] wrote {OUT}")


if __name__ == "__main__":
    main()
