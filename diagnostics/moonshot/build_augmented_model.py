"""M6 — Enlarge the shape space from the registrations themselves (one co-registration round).

THE ARGUMENT
M4b measured that pose + shape alone reaches fscore@0.02 = 0.731 with EXACTLY zero mesh
distortion, against the baseline's 0.884 bought with 74% edge distortion and the loss of 36%
of local vertex neighbourhoods. The residual 0.153 of F-score is what the free-form offsets
supply -- i.e. shape variation the 13-direction basis cannot express.

The literature's answer to "the shape space is too small" is not to tune the loss. It is to
rebuild the shape space from the registrations and iterate: Hirshberg et al., *Coregistration:
Simultaneous Alignment and Modeling of Articulated 3D Shape*, ECCV 2012; Zuffi et al. (SMAL,
CVPR 2017) run 4 such rounds; Bogo et al. (FAUST/Dyna) the same idea.

This runs ONE round of that loop:
  1. take a run whose offsets are in template correspondence (every fit shares the template's
     vertex indexing, so the deform_verts fields are directly comparable across specimens)
  2. PCA the offset fields -> new shape directions
  3. append them to a COPY of SMIL_OmniAnt.pkl as extra `shapedirs` columns
  4. refit with pose + shape ONLY, no free-form offsets at all

HONESTY REQUIREMENT
The new directions are learned from the very specimens they will be tested on, so a naive
version would be guaranteed to look good and mean nothing. This script therefore splits the
50 specimens into a TRAIN half (components learned here) and a TEST half (never seen), and the
report must quote the TEST half. `--split` controls this; `--split none` builds the
in-sample version for reference only.

Outputs a new .pkl next to the original, plus a JSON describing the spectrum.
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", default=os.path.join(os.path.dirname(__file__), "runs/A4_nofreeze"))
    ap.add_argument("--n_new", type=int, default=24, help="new shape directions to append")
    ap.add_argument("--split", choices=["train", "none"], default="train")
    ap.add_argument("--out_pkl", default=os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt_aug.pkl"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    npzs = sorted(
        [
            p
            for p in glob.glob(os.path.join(args.source_run, "*.npz"))
            if "_batch_" not in p and "start_selection" not in p
        ]
    )
    d = np.load(npzs[-1], allow_pickle=True)
    dv = d["deform_verts"].astype(np.float64)  # (B, V, 3) -- in template correspondence
    B, V, _ = dv.shape
    labels = [str(x) for x in d["labels"]]
    print(f"source: {npzs[-1]}   {B} specimens x {V} verts")

    idx = np.arange(B)
    if args.split == "train":
        train = idx[idx % 2 == 0]
        test = idx[idx % 2 == 1]
    else:
        train, test = idx, idx
    print(f"train specimens: {len(train)}   test specimens: {len(test)}  (components learned on TRAIN only)")
    np.savez(
        os.path.join(OUT, "m6_split.npz"),
        train=train,
        test=test,
        train_labels=np.array([labels[i] for i in train]),
        test_labels=np.array([labels[i] for i in test]),
    )

    X = dv[train].reshape(len(train), -1)  # (Ntrain, V*3)
    mean = X.mean(0)
    Xc = X - mean
    # economy SVD: Ntrain << V*3, so at most Ntrain-1 meaningful directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    cum = np.cumsum(var) / var.sum()
    k = min(args.n_new, Vt.shape[0])
    print(f"\noffset-field PCA: {k} of {Vt.shape[0]} available directions retained")
    print(f"  variance explained by {k} components: {100 * cum[k - 1]:.1f}%")
    for i in [0, 1, 2, 4, 9, min(19, len(cum) - 1)]:
        if i < len(cum):
            print(f"    first {i + 1:2d} components: {100 * cum[i]:5.1f}%")

    new_dirs = Vt[:k].reshape(k, V, 3).transpose(1, 2, 0)  # (V,3,k)

    # scale each new direction so that beta ~ N(0,1) produces a realistic offset, matching
    # the convention of the existing shapedirs (whose betas carry the model's shape_cov)
    scales = S[:k] / np.sqrt(max(len(train) - 1, 1))
    new_dirs = new_dirs * scales[None, None, :]

    with open(os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()

    old_sd = np.asarray(dd["shapedirs"], dtype=np.float64)  # (V,3,13)
    n_old = old_sd.shape[2]
    dd["shapedirs"] = np.concatenate([old_sd, new_dirs], axis=2)

    # the mean offset is a constant correction, so fold it into the template itself
    dd["v_template"] = np.asarray(dd["v_template"], dtype=np.float64) + mean.reshape(V, 3)

    # extend the shape prior: unit variance for the new directions (they are already scaled
    # so that beta ~ N(0,1) reproduces the training spread)
    old_cov = np.asarray(dd["shape_cov"], dtype=np.float64)
    n_tot = n_old + k
    cov = np.eye(n_tot)
    cov[:n_old, :n_old] = old_cov
    dd["shape_cov"] = cov
    dd["shape_mean_betas"] = np.concatenate([np.asarray(dd["shape_mean_betas"], dtype=np.float64), np.zeros(k)])
    # scaledirs / transdirs are indexed by beta and must grow to match, or the SMAL forward
    # pass will broadcast-error. New directions get zero joint scale/translation effect.
    for key in ("scaledirs", "transdirs"):
        if key in dd:
            a = np.asarray(dd[key], dtype=np.float64)  # (13, 55, 3)
            dd[key] = np.concatenate([a, np.zeros((k,) + a.shape[1:])], axis=0)

    with open(args.out_pkl, "wb") as f:
        pickle.dump(dd, f, protocol=2)
    print(f"\nwrote {args.out_pkl}")
    print(f"  shapedirs {old_sd.shape} -> {dd['shapedirs'].shape}")
    print(f"  shape_cov {old_cov.shape} -> {cov.shape}")
    for key in ("scaledirs", "transdirs"):
        if key in dd:
            print(f"  {key} -> {np.asarray(dd[key]).shape}")

    json.dump(
        dict(
            source=npzs[-1],
            n_new=int(k),
            n_old=int(n_old),
            var_explained=float(cum[k - 1]),
            spectrum=[float(x) for x in (var / var.sum())[:40]],
            train=[int(x) for x in train],
            test=[int(x) for x in test],
        ),
        open(os.path.join(OUT, "m6_augmented_model.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
