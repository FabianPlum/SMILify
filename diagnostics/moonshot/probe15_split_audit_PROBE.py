"""PROBE (verification): re-score probe 15's gates per specimen and break them down by the
35/15 train/held-out split written by make_partfield_data.py.

Faithful replication of probe_15_partfield_gates.py's per-specimen loop (same N_REF=30000,
same predict_field votes, same G1/G2/G3b/G3c definitions), on CPU, sharded over processes.
Writes one JSON shard per worker; merge with --merge.
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

DATA = os.path.join(HERE, "partfield")
OUTDIR = os.path.join(HERE, "out", "probe15_split_audit_PROBE")
N_REF = 30000
MIN_LEG_PTS = 50


def mirror_perm(names):
    idx = {n: i for i, n in enumerate(names)}
    return torch.tensor(
        [idx[n[:-2] + ("_r" if n.endswith("_l") else "_l")] if n[-2:] in ("_l", "_r") else idx[n] for n in names]
    )


def largest_cc_frac(pts, k=8):
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


def template_convention(names):
    import pickle

    with open(config.SMAL_FILE, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    from fitter_3d.partfield import template_vertex_labels

    vt = np.asarray(dd["v_template"], dtype=np.float64)
    vl = template_vertex_labels(dd["weights"], [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]])
    vt = vt - vt.mean(0)
    legx = {
        lg: float(np.mean([vt[vl == i, 0].mean() for i in [j for j, n in enumerate(names) if n.startswith(lg)]]))
        for lg in ("l1", "l2", "l3")
    }
    legy = {
        sd: float(np.mean(vt[np.isin(vl, [j for j, n in enumerate(names) if n.endswith("_" + sd)]), 1]))
        for sd in ("l", "r")
    }
    return legx["l1"] > legx["l2"] > legx["l3"], legy["l"] > legy["r"]


def score_one(net, names, mp, leg_ids, debris_id, x_desc, y_l_pos, mesh, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        a = sample_points_from_meshes(mesh, N_REF)[0]
        b = sample_points_from_meshes(mesh, N_REF)[0]
    an, _, _ = normalise(a)
    bn, _, _ = normalise(b)
    la = predict_field(net, an, seed=1).argmax(-1)
    lb = predict_field(net, bn, seed=2).argmax(-1)

    j = knn_points(an.unsqueeze(0), bn.unsqueeze(0), K=1).idx[0, :, 0]
    g1 = float((la == lb[j]).float().mean())

    mir = an * torch.tensor([1.0, -1.0, 1.0])
    lm = predict_field(net, mir, seed=1).argmax(-1)
    g2 = float((mp[lm] == la).float().mean())

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
            for u_, v_ in ((0, 1), (1, 2)):
                ok += int((xs[u_] > xs[v_]) if x_desc else (xs[u_] < xs[v_]))
                tot += 1
        for n in (1, 2, 3):
            dl, dr = cent[f"l{n}_l"][1], cent[f"l{n}_r"][1]
            ok += int((dl > dr) == y_l_pos)
            tot += 1
    order = ok / tot if tot else float("nan")
    return dict(
        g1=g1,
        g2=g2,
        six_legs=bool(six),
        order=order,
        coherence=float(np.mean(list(coh.values()))) if coh else float("nan"),
        debris_frac=float((lab == debris_id).mean()),
        leg_pts={k: int(v) for k, v in present.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(DATA, "net_B_mirror.pt"))
    ap.add_argument("--mesh_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--tag", default="B_mirror")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(f for f in os.listdir(args.mesh_dir) if f.endswith(".obj"))
    split = json.load(open(os.path.join(DATA, "split.json")))
    is_train = {f: (f in set(split["train"])) for f in files}

    if args.merge:
        rows = []
        for s in range(args.nshard):
            p = os.path.join(OUTDIR, f"{args.tag}_shard{s}.json")
            rows += json.load(open(p))
        rows.sort(key=lambda r: r["file"])
        json.dump(rows, open(os.path.join(OUTDIR, f"{args.tag}_all.json"), "w"), indent=1)
        print(f"merged {len(rows)} rows -> {args.tag}_all.json")
        return

    torch.set_num_threads(args.threads)
    ck = torch.load(args.net, map_location="cpu")
    names = ck["names"]
    net = PartFieldNet(n_classes=len(names), width=ck.get("width", 1.0))
    net.load_state_dict(ck["state"])
    net.eval()
    mp = mirror_perm(names)
    debris_id = names.index("debris")
    leg_ids = {}
    for lg in ("l1", "l2", "l3"):
        for sd in ("l", "r"):
            leg_ids[f"{lg}_{sd}"] = [i for i, n in enumerate(names) if n.startswith(lg) and n.endswith("_" + sd)]
    x_desc, y_l_pos = template_convention(names)

    rows = []
    for i, f in enumerate(files):
        if i % args.nshard != args.shard:
            continue
        _, m = load_meshes(mesh_files=[os.path.join(args.mesh_dir, f)], device="cpu")
        r = score_one(net, names, mp, leg_ids, debris_id, x_desc, y_l_pos, m, seed=1000 + i)
        r["file"] = f
        r["name"] = f.replace("_processed.obj", "")
        r["split"] = "train" if is_train[f] else "test"
        rows.append(r)
        print(
            f"[{args.shard}] {i} {r['split']:5s} g1={r['g1']:.3f} g2={r['g2']:.3f} ord={r['order']:.2f} {r['name'][:40]}",
            flush=True,
        )
    json.dump(rows, open(os.path.join(OUTDIR, f"{args.tag}_shard{args.shard}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
