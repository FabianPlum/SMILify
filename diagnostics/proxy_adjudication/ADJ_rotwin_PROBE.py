"""ADJUDICATION PROBE A -- is volume IoU's ONLY win (small-angle rotation of a big blob) real,
or is it an artifact of chamfer being RESAMPLED every evaluation?

GEOMRISK_PREMISE_PROBE draws 4000 fresh surface samples per repeat, so chamfer's sd is a
resampling floor that shrinks as 1/sqrt(N) and vanishes entirely if the sample set is FROZEN
(which is what a real optimiser can trivially do -- fixed target point cloud + fixed template
sample indices). Volume IoU's MC noise is fixable the same way (frozen query points).

Re-measure detectability at small rotation under three chamfer estimators:
  resampled  (as in the original probe)
  frozen     (same sample points at every delta -- what an optimiser actually uses)
and volume IoU with frozen query points too, for a like-for-like comparison.
"""
import os, sys, json, pickle
import numpy as np, trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
N_SURF, N_MC, N_REP = 4000, 20000, 12

def part_submesh(v, f, keep):
    fm = keep[f].all(1); idx = -np.ones(len(v), np.int64)
    sel = np.unique(f[fm]); idx[sel] = np.arange(len(sel))
    return trimesh.Trimesh(v[sel], idx[f[fm]], process=False)

def main():
    with open(os.path.join(REPO, __import__("config").SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    jn = [str(s) for s in dd["J_names"]]
    v = np.asarray(dd["v_template"], np.float64); f = np.asarray(dd["f"]).astype(np.int64)
    dom = np.asarray(dd["weights"]).argmax(1)
    c0 = v.mean(0); v = (v - c0) / np.abs(v - c0).max()
    parts = {"gaster":[j for j in range(len(jn)) if jn[j].startswith("b_a_")],
             "mesosoma":[j for j in range(len(jn)) if jn[j]=="b_t"],
             "head":[j for j in range(len(jn)) if jn[j]=="b_h"]}
    out = {}
    for pname, js in parts.items():
        sub = part_submesh(v, f, np.isin(dom, js))
        if not sub.is_watertight: sub.fill_holes()
        pts = np.asarray(sub.vertices); c = pts.mean(0)
        _, S, V = np.linalg.svd(pts - c, full_matrices=False)
        axis = V[0]; L = 2*np.percentile(np.abs((pts-c)@V[0]), 98)
        lo, hi = sub.bounds[0]-0.15*L, sub.bounds[1]+0.15*L

        # FROZEN sample sets: one target cloud, one model cloud, one query set -- reused at every delta
        rngf = np.random.default_rng(7)
        tgt_pts = trimesh.sample.sample_surface(sub, N_SURF, seed=11)[0]
        mdl_bary = trimesh.sample.sample_surface(sub, N_SURF, seed=12)[0]
        qry = lo + rngf.random((N_MC,3))*(hi-lo)
        tgt_tree = cKDTree(tgt_pts); tgt_in = sub.contains(qry)

        rows = []
        for deg in [1.0, 2.0, 4.0, 8.0, 16.0]:
            R = trimesh.transformations.rotation_matrix(np.deg2rad(deg), axis, c)[:3,:3]
            mov = sub.copy(); mov.vertices = (np.asarray(sub.vertices)-c)@R.T + c
            # --- resampled chamfer (original probe's estimator)
            rs = []
            for r in range(N_REP):
                rr = np.random.default_rng(1000+r)
                pa = trimesh.sample.sample_surface(mov, N_SURF, seed=int(rr.integers(1<<30)))[0]
                pb = trimesh.sample.sample_surface(sub, N_SURF, seed=int(rr.integers(1<<30)))[0]
                rs.append(cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())
            base_rs = []
            for r in range(N_REP):
                rr = np.random.default_rng(1000+r)
                pa = trimesh.sample.sample_surface(sub, N_SURF, seed=int(rr.integers(1<<30)))[0]
                pb = trimesh.sample.sample_surface(sub, N_SURF, seed=int(rr.integers(1<<30)))[0]
                base_rs.append(cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())
            sd_rs = float(np.std(base_rs + rs))
            det_rs = abs(np.mean(rs)-np.mean(base_rs))/max(sd_rs,1e-12)
            # --- FROZEN chamfer: model points ride rigidly with the part, target cloud fixed
            mp = (mdl_bary - c)@R.T + c
            ch_fr = float(tgt_tree.query(mp)[0].mean() + cKDTree(mp).query(tgt_pts)[0].mean())
            ch_fr0 = float(tgt_tree.query(mdl_bary)[0].mean() + cKDTree(mdl_bary).query(tgt_pts)[0].mean())
            # frozen chamfer noise: only from WHICH points were frozen -> estimate across 12 draws of the frozen set
            fr_noise = []
            for r in range(N_REP):
                a = trimesh.sample.sample_surface(sub, N_SURF, seed=200+r)[0]
                b = trimesh.sample.sample_surface(sub, N_SURF, seed=900+r)[0]
                fr_noise.append(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())
            sd_fr = float(np.std(fr_noise))
            det_fr = abs(ch_fr-ch_fr0)/max(sd_fr,1e-12)
            # --- volume IoU with FROZEN query points
            mv_in = mov.contains(qry)
            iou = float((tgt_in & mv_in).sum()/max((tgt_in | mv_in).sum(),1))
            p = float(tgt_in.mean())
            sd_iou = float(np.sqrt(max(p*(1-p),1e-9)/N_MC))*2
            det_iou = abs(iou-1.0)/max(sd_iou,1e-12)
            rows.append(dict(deg=deg, det_cham_resampled=det_rs, det_cham_frozen=det_fr,
                             det_viou=det_iou, iou=iou, dchamfer_frozen=ch_fr-ch_fr0))
        out[pname] = rows
        print(f"=== {pname}")
        print(f"{'deg':>5} {'det_cham_RESAMP':>16} {'det_cham_FROZEN':>16} {'det_viou':>10} {'iou':>8}")
        for r in rows:
            print(f"{r['deg']:5.1f} {r['det_cham_resampled']:16.1f} {r['det_cham_frozen']:16.1f} {r['det_viou']:10.1f} {r['iou']:8.4f}")
    json.dump(out, open(os.path.join(HERE,"adj_rotwin.json"),"w"), indent=1)

main()
