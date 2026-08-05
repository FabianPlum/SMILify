"""Multi-round co-registration over the full 757-worker corpus (item 4).

WHAT ROUND 1 ESTABLISHED (probe 11, M6)
Appending 24 PCA directions learned from the offset fields of 25 train specimens, then
refitting pose+shape only:
  * held-out distal legs -26.0% (19/25, p=0.015) and body -5.9% (19/25, p=0.015)
    -> the directions GENERALIZE; this is not memorisation (test improved MORE than train)
  * held-out fscore@0.02 -0.2% (ns)
    -> but they do NOT close the global surface gap
  * offset PCA spectrum is flat: component 1 explains 10.9%, 20 needed for 93.5%
    -> the residual is genuinely high-dimensional, so this needs SCALE, not cleverness
  * AND the resulting model produced badly FOLDED geometry: 2.79% of adjacent face pairs
    folded past 90 deg, vs 0.54% for the baseline (probe 12)

THE BLOCKER, AND WHY THIS SCRIPT IS NOT JUST "ROUND 1 WITH MORE DATA"
That last point is the one that matters. PCA directions fitted to displacement fields are
optimal in a least-squares sense and carry no guarantee that the mesh they generate is
valid. Driving them as linear blendshapes at magnitudes the PCA never saw produces
self-folding surfaces. Simply adding more specimens gives MORE such directions, so folding
would get worse, not better.

Three constraints are therefore applied here that round 1 did not have:

  1. FOLD-REJECTED DIRECTIONS. Every candidate direction is driven to +/-3 sigma on its own
     and the resulting mesh is checked for folded adjacent faces. A direction that folds the
     mesh in isolation is dropped, however much variance it explains. Variance is not the
     selection criterion; validity is a hard gate.

  2. SMOOTHNESS-REGULARISED BASIS. Candidate fields are Laplacian-smoothed before the PCA,
     so high-frequency per-vertex noise (which is where folding comes from) does not enter
     the basis at all. Controlled by --smooth_iters.

  3. BILATERAL SYMMETRISATION. The template's mirror involution is exact (verified: max
     match distance 0.00e+00, 100% involution), and ant SHAPE is bilaterally symmetric even
     though pose is not. Symmetrising each offset field before the PCA halves the effective
     noise and stops the basis from encoding pose asymmetry as if it were shape.

PROTOCOL
Each round: fit the corpus -> collect offsets -> build a constrained basis from the TRAIN
split only -> write an augmented model -> refit -> score on the HELD-OUT split. The train/
test split is fixed across all rounds (seeded), so round-over-round improvement on the test
split is a real generalisation curve and not a slow leak of test data into the basis.

Go/no-go per round, pre-registered: a round is kept only if, on the HELD-OUT split,
part_leg_distal improves AND folded_face_frac does not exceed the baseline's 0.54%.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "out")
RUNS = os.path.join(HERE, "runs")
BASELINE_FOLD_FRAC = 0.0054  # probe 12, stock pipeline


# ------------------------------------------------------------------ mesh helpers
def face_adjacency(faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e, _ = torch.sort(e, dim=1)
    fid = torch.arange(faces.shape[0], device=faces.device).repeat(3)
    k = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    o = torch.argsort(k)
    k, fid = k[o], fid[o]
    i = torch.nonzero(k[1:] == k[:-1], as_tuple=True)[0]
    return fid[i], fid[i + 1]


def folded_fraction(verts, faces, a, b):
    """Fraction of adjacent face pairs whose normals disagree by more than 90 degrees."""
    t = verts[faces]
    n = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    cos = (n[a] * n[b]).sum(-1).clamp(-1, 1)
    return float((cos < 0.0).float().mean())


def build_laplacian(verts_n, faces):
    """Uniform graph Laplacian as a sparse operator, for smoothing offset fields."""
    V = verts_n
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    idx = e.t()
    val = torch.ones(idx.shape[1], device=faces.device)
    A = torch.sparse_coo_tensor(idx, val, (V, V)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
    return A, deg


def smooth_field(field, A, deg, iters):
    """Jacobi-smooth a per-vertex vector field: x <- (x + A x / deg) / 2."""
    x = field
    for _ in range(iters):
        x = 0.5 * (x + torch.sparse.mm(A, x) / deg.unsqueeze(-1))
    return x


# ------------------------------------------------------------------ basis construction
def build_constrained_basis(offsets, v_template, faces, sym_map, n_keep, smooth_iters, device):
    """PCA over offset fields with symmetrisation, smoothing, and a hard fold gate.

    offsets: (N, V, 3) free-form offset fields from the previous round, TRAIN split only.
    Returns (directions (V,3,k), scales (k,), mean (V,3), report dict).
    """
    N, V, _ = offsets.shape
    X = torch.tensor(offsets, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.int64, device=device)
    vt = torch.tensor(v_template, dtype=torch.float32, device=device)
    a, b = face_adjacency(f)

    # 1. bilateral symmetrisation -- shape is symmetric even where pose is not
    m = torch.tensor(sym_map, dtype=torch.long, device=device)
    mirror = X[:, m, :].clone()
    mirror[:, :, 1] *= -1
    X = 0.5 * (X + mirror)

    # 2. Laplacian smoothing -- keeps high-frequency noise out of the basis, which is
    #    where the folding in round 1 came from
    if smooth_iters > 0:
        A, deg = build_laplacian(V, f)
        X = torch.stack([smooth_field(X[i], A, deg, smooth_iters) for i in range(N)])

    mean = X.mean(0)
    Xc = (X - mean).reshape(N, -1)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    var = S**2
    cum = torch.cumsum(var, 0) / var.sum()

    # 3. fold gate -- drive each candidate to +/-3 sigma alone and reject if it folds
    kept, rejected = [], []
    scales_all = S / np.sqrt(max(N - 1, 1))
    base_fold = folded_fraction(vt + mean, f, a, b)
    for i in range(min(Vt.shape[0], n_keep * 3)):
        d = Vt[i].reshape(V, 3) * scales_all[i]
        worst = max(folded_fraction(vt + mean + s * 3.0 * d, f, a, b) for s in (-1.0, 1.0))
        if worst <= max(BASELINE_FOLD_FRAC, base_fold * 1.05):
            kept.append(i)
        else:
            rejected.append((i, worst))
        if len(kept) >= n_keep:
            break

    rep = dict(
        n_candidates=int(Vt.shape[0]),
        n_kept=len(kept),
        n_rejected=len(rejected),
        base_fold=base_fold,
        var_explained_by_kept=float(cum[kept[-1]]) if kept else 0.0,
        rejected_worst_fold=[float(w) for _, w in rejected[:10]],
    )
    if not kept:
        return None, None, mean.cpu().numpy(), rep

    dirs = torch.stack([Vt[i].reshape(V, 3) * scales_all[i] for i in kept], -1)
    return dirs.cpu().numpy(), scales_all[kept].cpu().numpy(), mean.cpu().numpy(), rep


def write_model(base_pkl, out_pkl, dirs, mean):
    with open(base_pkl, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    old = np.asarray(dd["shapedirs"], dtype=np.float64)
    n_old, k = old.shape[2], dirs.shape[2]
    dd["shapedirs"] = np.concatenate([old, dirs.astype(np.float64)], axis=2)
    dd["v_template"] = np.asarray(dd["v_template"], dtype=np.float64) + mean
    cov = np.eye(n_old + k)
    cov[:n_old, :n_old] = np.asarray(dd["shape_cov"], dtype=np.float64)
    dd["shape_cov"] = cov
    dd["shape_mean_betas"] = np.concatenate([np.asarray(dd["shape_mean_betas"], dtype=np.float64), np.zeros(k)])
    for key in ("scaledirs", "transdirs"):
        if key in dd:
            arr = np.asarray(dd[key], dtype=np.float64)
            dd[key] = np.concatenate([arr, np.zeros((k,) + arr.shape[1:])], axis=0)
    with open(out_pkl, "wb") as fh:
        pickle.dump(dd, fh, protocol=2)
    return dd["shapedirs"].shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_dir", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument(
        "--n_specimens",
        type=int,
        default=400,
        help="corpus size per round; 757 available. Kept below the full set by "
        "default so a round fits in memory as one batch group.",
    )
    ap.add_argument("--n_new", type=int, default=32, help="directions to KEEP per round (post fold-gate)")
    ap.add_argument("--smooth_iters", type=int, default=6)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.worker_dir, "*.obj")))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    files = files[: args.n_specimens]
    n_train = len(files) // 2
    train_files, test_files = files[:n_train], files[n_train:]
    print(f"[coreg] corpus {len(files)}  train {len(train_files)}  test {len(test_files)}", flush=True)
    print("[coreg] split is FIXED across rounds, so test-split gains are a real generalisation curve", flush=True)

    # staging dirs of symlinks, so the existing runners work unchanged
    for name, fl in (("coreg_train", train_files), ("coreg_test", test_files)):
        d = os.path.join(HERE, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for p in fl:
            os.symlink(p, os.path.join(d, os.path.basename(p)))

    print("\n[coreg] NOTE: this script stages the corpus and documents the protocol; the")
    print("[coreg] per-round fit/collect/rebuild loop is driven by run_next_batch.sh coreg.")
    print("[coreg] The fold gate and the symmetrised, smoothed basis are implemented above")
    print("[coreg] and are the part that round 1 lacked -- see module docstring.\n")

    json.dump(
        dict(
            train=[os.path.basename(p) for p in train_files],
            test=[os.path.basename(p) for p in test_files],
            seed=args.seed,
            rounds=args.rounds,
            n_new=args.n_new,
            smooth_iters=args.smooth_iters,
        ),
        open(os.path.join(OUT, "coreg_split.json"), "w"),
        indent=1,
    )
    print(f"[coreg] wrote {os.path.join(OUT, 'coreg_split.json')}")


if __name__ == "__main__":
    main()
