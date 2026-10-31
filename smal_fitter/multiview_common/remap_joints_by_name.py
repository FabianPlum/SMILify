"""Remap a 40-joint SLEAP multi-view HDF5 to the 51-joint v6 model (v5 + proxy leaf bones), by joint name.

The source must already be a copy of the training HDF5 (`cp` it first). Only the datasets with a joint axis are
rewritten; images, cameras and alignment data stay byte-identical, so the silhouette sidecar built on the source
still pairs with it. Slots follow the v6 J_names order (the add-on's depth-first bone order):
- every v5 joint keeps its data under its name, except the label slots the convention moves to a proxy;
- each moved label goes to its proxy slot (M103: hip_center -> hip_center_m103, shoulders -> shoulder_*_m103,
  knees -> knee_*_m103), and its old slot becomes unlabelled;
- unlabelled = the source's own fill for never-labelled joints (zeros, visibility 0, confidence 0).

  python -m smal_fitter.multiview_common.remap_joints_by_name <copy.h5> <v5.pkl> <v6.pkl> <convention>
"""

import json
import pickle
import sys

import h5py
import numpy as np

path, v5p, v6p, conv = sys.argv[1:5]
n5 = list(pickle.load(open(v5p, "rb"), encoding="latin1")["J_names"])
n6 = list(pickle.load(open(v6p, "rb"), encoding="latin1")["J_names"])
MOVES = {
    "m103": {"Lumbar-Vertebrae": "hip_center_m103", "humerus_L": "shoulder_L_m103", "humerus_R": "shoulder_R_m103",
             "tibia_L": "knee_L_m103", "tibia_R": "knee_R_m103"},
    "falkner": {"skull": "head_falkner", "Tail_06": "Tail_06_falkner", "tibia_L": "knee_L_falkner",
                "tibia_R": "knee_R_falkner", "Foot_L": "heel_L_falkner", "Foot_R": "heel_R_falkner"},
}[conv]
# source slot (v5 index) for each v6 slot, -1 = unlabelled
src = []
for n in n6:
    moved_from = [k for k, v in MOVES.items() if v == n]
    if moved_from:
        src.append(n5.index(moved_from[0]))
    elif n in n5 and n not in MOVES:
        src.append(n5.index(n))
    else:
        src.append(-1)
src = np.array(src)
# datasets with a joint axis: (name, axis)
JOINT_DS = [("multiview_keypoints/keypoints_2d", 2), ("multiview_keypoints/keypoints_2d_raw", 2),
            ("multiview_keypoints/keypoint_visibility", 2), ("multiview_keypoints/keypoint_visibility_raw", 2),
            ("multiview_keypoints/keypoint_confidence_raw", 2), ("multiview_keypoints/keypoints_3d", 1),
            ("parameters/joint_rot", 1)]
with h5py.File(path, "r+") as f:
    md = f["metadata"].attrs
    assert int(md["n_joints"]) == len(n5), f"source has {md['n_joints']} joints, v5 has {len(n5)}"
    assert "joint_names" not in md or json.loads(md["joint_names"]) == n5
    for name, ax in JOINT_DS:
        d = f[name]
        a = d[()]
        assert a.shape[ax] == len(n5), (name, a.shape)
        out = np.zeros(a.shape[:ax] + (len(n6),) + a.shape[ax + 1 :], a.dtype)
        idx = [slice(None)] * a.ndim
        for j, s in enumerate(src):
            if s >= 0:
                idx_o, idx_s = list(idx), list(idx)
                idx_o[ax], idx_s[ax] = j, s
                out[tuple(idx_o)] = a[tuple(idx_s)]
        chunks = list(d.chunks)
        comp, opts = d.compression, d.compression_opts
        del f[name]
        f.create_dataset(name, data=out, chunks=tuple(chunks), compression=comp, compression_opts=opts)
        print(f"[remap] {name}: {a.shape} -> {out.shape}", flush=True)
    md["n_joints"] = str(len(n6)) if isinstance(md["n_joints"], str) else len(n6)
    md["n_pose"] = str(len(n6) - 1) if isinstance(md["n_pose"], str) else len(n6) - 1
    md["joint_names"] = json.dumps(n6)
    md["proxy_remap"] = json.dumps({"convention": conv, "moves": MOVES, "v5_pkl": v5p, "v6_pkl": v6p,
                                    "source_slot_per_joint": src.tolist()})
print("[remap] done:", path)
