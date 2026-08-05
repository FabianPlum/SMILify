"""PROBE 15 — score the part field against the gates pre-registered in fitter_3d/partfield.py.

The gates were fixed before any part-field result existed, in the module docstring, and are
reproduced here verbatim. Three of the four need no ground truth at all, which is the point:
they test properties the network must have to be usable, not agreement with labels that are
themselves fit-derived.

  G1  REPRODUCIBILITY >= 90%.  Two independent surface samplings of the same scan must
      receive the same labels. This is the gate that killed the geodesic level-set method
      (20% agreement on branch count, probe 14).
  G2  BILATERAL CONSISTENCY >= 85%.  Mirroring the input in y must swap left/right labels
      and preserve segment identity.
  G3  TOUCHING-LEG RECOVERY >= 80%.  On the specimens where the geodesic method collapsed
      to <= 2 branches, all six legs must still be recovered.
  G4  HELD-OUT SYNTHETIC ACCURACY >= 85%.

G3 as originally written is too weak on its own -- a network that painted six arbitrary
blobs would pass it -- so it is scored together with two unsupervised structural checks
that a blob-painter fails:

  G3b ANATOMICAL ORDERING.  The predicted leg centroids must be correctly ordered along the
      body axis (pro/meso/meta) and correctly separated in y (left/right). Eight ordering
      constraints per specimen, all checkable with no labels whatsoever. This is a genuine
      test of whether the field means anything.
  G3c SPATIAL COHERENCE.  Each predicted leg must be one connected cluster, not scatter:
      largest connected component / total points for that label.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.partfield import PartFieldNet, predict_field, normalise  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "partfield")
GATES = dict(G1=0.90, G2=0.85, G3=0.80, G4=0.85)
N_REF = 30000
MIN_LEG_PTS = 50


def load_net(path, dev):
    ck = torch.load(path, map_location=dev)
    net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["names"], ck


def mirror_perm(names):
    idx = {n: i for i, n in enumerate(names)}
    return torch.tensor([idx[n[:-2] + ("_r" if n.endswith("_l") else "_l")]
                         if n[-2:] in ("_l", "_r") else idx[n] for n in names])


def largest_cc_frac(pts, k=8):
    """Fraction of a point set in its largest connected component of a kNN graph."""
    n = len(pts)
    if n < 4:
        return float("nan")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    kk = min(k, n - 1)
    d, idx = cKDTree(pts).query(pts, k=kk + 1)
    rows = np.repeat(np.arange(n), kk)
    A = coo_matrix((np.ones(n * kk), (rows, idx[:, 1:].ravel())), shape=(n, n))
    _, lab = connected_components(A.maximum(A.T), directed=False)
    return float(np.bincount(lab).max() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(DATA, "net_B_mirror.pt"))
    ap.add_argument("--mesh_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--tag", default="B_mirror")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    net, names, ck = load_net(args.net, dev)
    mp = mirror_perm(names).to(dev)
    debris_id = names.index("debris")
    leg_ids = {}
    for lg in ("l1", "l2", "l3"):
        for sd in ("l", "r"):
            leg_ids[f"{lg}_{sd}"] = [i for i, n in enumerate(names)
                                     if n.startswith(lg) and n.endswith("_" + sd)]

    print(f"PROBE 15 — part-field gates, model '{args.tag}'")
    print(f"  trained: acc {ck['acc']:.4f}  mIoU {ck['miou']:.4f}  distal-IoU {ck['distal_iou']:.4f}"
          f"  (epoch {ck['epoch']})")
    print(f"  args: {json.dumps({k: v for k, v in ck['args'].items() if k not in ('out',)})}\n")

    # -------------------------------------------------------------- body-axis convention
    # Determined from the TEMPLATE, not assumed: which sign of x is anterior, and does leg
    # index increase forwards or backwards?
    import pickle
    with open(config.SMAL_FILE, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    from fitter_3d.partfield import template_vertex_labels
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    vl = template_vertex_labels(dd["weights"], [n.decode() if isinstance(n, bytes) else n
                                                for n in dd["J_names"]])
    vt = vt - vt.mean(0)
    legx = {lg: float(np.mean([vt[vl == i, 0].mean() for i in
                              [j for j, n in enumerate(names) if n.startswith(lg)]]))
            for lg in ("l1", "l2", "l3")}
    legy = {sd: float(np.mean(vt[np.isin(vl, [j for j, n in enumerate(names)
                                              if n.endswith("_" + sd)]), 1]))
            for sd in ("l", "r")}
    x_desc = legx["l1"] > legx["l2"] > legx["l3"]
    x_asc = legx["l1"] < legx["l2"] < legx["l3"]
    y_l_pos = legy["l"] > legy["r"]
    print(f"  template convention: leg mean-x l1={legx['l1']:+.4f} l2={legx['l2']:+.4f} "
          f"l3={legx['l3']:+.4f} -> {'l1 anterior (+x)' if x_desc else 'l1 posterior (-x)'}")
    print(f"                       mean-y  _l={legy['l']:+.4f} _r={legy['r']:+.4f}\n")
    assert x_desc or x_asc, "template leg order along x is not monotone"

    # -------------------------------------------------------------- per-specimen scoring
    _, meshes = load_meshes(mesh_dir=args.mesh_dir, sorting=sorted, device=str(dev))
    files = sorted(f for f in os.listdir(args.mesh_dir) if f.endswith(".obj"))
    rows = []
    for i, f in enumerate(files):
        with torch.no_grad():
            a = sample_points_from_meshes(meshes[i], N_REF)[0]
            b = sample_points_from_meshes(meshes[i], N_REF)[0]
        an, _, _ = normalise(a)
        bn, _, _ = normalise(b)
        la = predict_field(net, an, seed=1).argmax(-1)
        lb = predict_field(net, bn, seed=2).argmax(-1)

        # G1: agreement between two independent samplings, matched by nearest neighbour
        j = knn_points(an.unsqueeze(0), bn.unsqueeze(0), K=1).idx[0, :, 0]
        g1 = float((la == lb[j]).float().mean())

        # G2: mirror the SAME points, so no matching error enters the number
        mir = an * torch.tensor([1.0, -1.0, 1.0], device=dev)
        lm = predict_field(net, mir, seed=1).argmax(-1)
        g2 = float((mp[lm] == la).float().mean())

        # G3 / G3b / G3c
        lab = la.cpu().numpy()
        pts = an.cpu().numpy()
        present, cent, coh = {}, {}, {}
        for k, ids in leg_ids.items():
            m = np.isin(lab, ids)
            present[k] = int(m.sum())
            if m.sum() >= MIN_LEG_PTS:
                cent[k] = pts[m].mean(0)
                coh[k] = largest_cc_frac(pts[m])
        six = all(v >= MIN_LEG_PTS for v in present.values())

        ok = tot = 0
        if len(cent) == 6:
            for sd in ("l", "r"):
                xs = [cent[f"l{n}_{sd}"][0] for n in (1, 2, 3)]
                for u, v in ((0, 1), (1, 2)):
                    ok += int((xs[u] > xs[v]) if x_desc else (xs[u] < xs[v]))
                    tot += 1
            for n in (1, 2, 3):
                dl, dr = cent[f"l{n}_l"][1], cent[f"l{n}_r"][1]
                ok += int((dl > dr) == y_l_pos)
                tot += 1
        order = ok / tot if tot else float("nan")

        rows.append(dict(name=f.replace("_processed.obj", ""), g1=g1, g2=g2,
                         six_legs=bool(six), order=order,
                         coherence=float(np.mean(list(coh.values()))) if coh else float("nan"),
                         debris_frac=float((lab == debris_id).mean()),
                         leg_pts={k: int(v) for k, v in present.items()}))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(files)}", flush=True)

    g1 = np.array([r["g1"] for r in rows])
    g2 = np.array([r["g2"] for r in rows])
    six = np.array([r["six_legs"] for r in rows])
    order = np.array([r["order"] for r in rows], dtype=float)
    coh = np.array([r["coherence"] for r in rows], dtype=float)

    # -------------------------------------------------------------- G3 subset from probe 14
    p14 = json.load(open(os.path.join(OUT, "probe14_topology_gate.json")))
    hard = {r["name"] for r in p14["rows"] if r["best_seed0"] <= 2}
    sub = np.array([r["name"] in hard for r in rows])
    g3 = float(six[sub].mean()) if sub.any() else float("nan")

    # -------------------------------------------------------------- G4
    va = np.load(os.path.join(DATA, "synth_val.npz"))
    X = torch.tensor(va["pts"]).to(dev)
    Y = torch.tensor(va["lab"]).to(dev)
    with torch.no_grad():
        pr = torch.cat([net(X[k:k + 16]).argmax(-1) for k in range(0, X.shape[0], 16)])
    g4 = float((pr == Y).float().mean())

    print("\n" + "=" * 86)
    print("PRE-REGISTERED GATES")
    print("=" * 86)
    res = {}
    for k, val, txt in (
        ("G1", float(g1.mean()), "reproducibility across two surface samplings"),
        ("G2", float(g2.mean()), "bilateral consistency under a mirror"),
        ("G3", g3, f"all six legs on the {int(sub.sum())} specimens the geodesic method lost"),
        ("G4", g4, "held-out synthetic per-point accuracy"),
    ):
        p = val >= GATES[k]
        res[k] = dict(value=val, gate=GATES[k], pass_=bool(p))
        print(f"  {k}  {100*val:6.2f}%   gate {100*GATES[k]:.0f}%   "
              f"{'PASS' if p else 'FAIL'}   {txt}")
    print(f"\n  G3b anatomical ordering satisfied: {100*np.nanmean(order):6.2f}% of 8 "
          f"constraints/specimen  ({int((order == 1).sum())}/{len(order)} specimens perfect)")
    print(f"  G3c spatial coherence (largest CC / leg): {100*np.nanmean(coh):6.2f}%")
    print(f"\n  reference: the geodesic level-set method (probe 14) scored 20% on the "
          f"G1-equivalent\n             and reached 6 branches on {100*p14['frac_ge6']:.0f}% of "
          f"specimens against the same 80% gate.")

    worst = np.argsort(g1)[:5]
    print("\n  least reproducible specimens:")
    for i in worst:
        print(f"    G1 {100*g1[i]:5.1f}%  G2 {100*g2[i]:5.1f}%  order {order[i]:.2f}  "
              f"{rows[i]['name'][:48]}")

    json.dump(dict(tag=args.tag, gates=res, order=float(np.nanmean(order)),
                   coherence=float(np.nanmean(coh)), rows=rows,
                   trained=dict(acc=ck["acc"], miou=ck["miou"], distal_iou=ck["distal_iou"])),
              open(os.path.join(OUT, f"probe15_gates_{args.tag}.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, f'probe15_gates_{args.tag}.json')}")


if __name__ == "__main__":
    main()
