"""PROBE 25 -- confirm the clean corpus REPEAT pair was fitted independently.

'ectatomma-tuberculatum.obj' and 'ectatomma-tuberculatum (1).obj' in
diagnostics/moonshot/clean81 are BYTE IDENTICAL (md5 d758d652435b080d0ad5a8bd1cd3cc4b),
so the ALL_ANTS_CLEAN fit contains the same scan fitted twice. Probe 22 uses that pair as
the repeatability floor for fitted leg proportions (0.63%). That floor only means anything
if the two rows actually took different optimisation paths rather than being a copy.
"""

import numpy as np

NPZ = "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
OUT = "/home/fabi/dev/SMILify/diagnostics/moonshot/joint_placement_repeat_pair_out.txt"


def main():
    d = np.load(NPZ, allow_pickle=True)
    lab = [str(x) for x in d["labels"]]
    i = lab.index("ectatomma-tuberculatum")
    j = lab.index("ectatomma-tuberculatum (1)")
    lines = [f"rows {i} {j}"]
    for k in ["betas", "log_beta_scales", "global_rot", "joint_rot", "trans", "deform_verts", "verts"]:
        a, b = d[k][i], d[k][j]
        rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-12)
        lines.append(f"  {k:16s} identical={np.array_equal(a, b)}  max|diff|={np.abs(a - b).max():.5g}  rel={rel:.4f}")
    v = d["verts"]
    diag = np.linalg.norm(v[i].max(0) - v[i].min(0))
    lines.append(f"  verts RMS diff / bbox diag = {np.sqrt(((v[i] - v[j]) ** 2).sum(-1)).mean() / diag:.5f}")
    txt = "\n".join(lines)
    print(txt)
    with open(OUT, "w") as f:
        f.write(__doc__ + "\n" + txt + "\n")


if __name__ == "__main__":
    main()
