#!/usr/bin/env python3
"""
Remap the per-joint arrays of a multi-view SLEAP HDF5 to another model's joint order, by joint name.

Reorders and extends every per-joint array of the HDF5 from the joint order of the source model (the model the HDF5
was built for) to the joint order of a target model, matching joints by name. A model that gains or reorders joints
needs this: the Blender add-on exports bones depth-first, so a new bone shifts the index of every joint after it.
Example: SMILy mouse v5 (40 joints) -> v6 (51 joints, v5 plus proxy leaf joints).

Modifies the HDF5 IN PLACE, so run it on a copy:

    python -m smal_fitter.multiview_common.remap_joints_by_name DATASET_COPY.h5 SOURCE.pkl TARGET.pkl CONVENTION

- A target joint that also exists in the source model keeps its data, unless the convention moves it.
- CONVENTION selects one table of label moves (MOVES): the labels of each listed source joint go to a proxy joint of
  the target model, and the source joint's own slot becomes unlabelled.
- Every other target joint is unlabelled: all zeros (keypoints, visibility, confidence), which is how the HDF5 stores
  joints that were never labelled.
- Only the per-joint datasets (JOINT_DATASETS, which must all exist) are rewritten, keeping their dtype, chunking and
  compression. Images, cameras and all other datasets are left untouched, so sidecars built on the source still pair
  with the copy. The metadata gets n_joints, n_pose and joint_names in the target order, plus proxy_remap: the
  convention, the moves, both model paths ("v5_pkl" = source, "v6_pkl" = target) and the source slot of every target
  joint (-1 = unlabelled).
"""

import argparse
import json
import pickle

import h5py
import numpy as np

# Labelling conventions: source joint -> proxy joint of the target model that takes over its labels.
MOVES = {
    "m103": {
        "Lumbar-Vertebrae": "hip_center_m103",
        "humerus_L": "shoulder_L_m103",
        "humerus_R": "shoulder_R_m103",
        "tibia_L": "knee_L_m103",
        "tibia_R": "knee_R_m103",
    },
    "falkner": {
        "skull": "head_falkner",
        "Tail_06": "Tail_06_falkner",
        "tibia_L": "knee_L_falkner",
        "tibia_R": "knee_R_falkner",
        "Foot_L": "heel_L_falkner",
        "Foot_R": "heel_R_falkner",
    },
}
# Datasets with a joint axis: (name, axis).
JOINT_DATASETS = [
    ("multiview_keypoints/keypoints_2d", 2),
    ("multiview_keypoints/keypoints_2d_raw", 2),
    ("multiview_keypoints/keypoint_visibility", 2),
    ("multiview_keypoints/keypoint_visibility_raw", 2),
    ("multiview_keypoints/keypoint_confidence_raw", 2),
    ("multiview_keypoints/keypoints_3d", 1),
    ("parameters/joint_rot", 1),
]


def _joint_names(pkl_path):
    with open(pkl_path, "rb") as f:
        return list(pickle.load(f, encoding="latin1")["J_names"])


def source_slots(source_names, target_names, moves):
    """Source joint index for every target joint, matched by name (-1 = unlabelled)."""
    src = []
    for n in target_names:
        moved_from = [k for k, v in moves.items() if v == n]
        if moved_from:
            src.append(source_names.index(moved_from[0]))
        elif n in source_names and n not in moves:
            src.append(source_names.index(n))
        else:
            src.append(-1)
    return np.array(src)


def remap_joints_by_name(path, source_pkl, target_pkl, convention):
    """Rewrite the per-joint datasets of the HDF5 at `path` IN PLACE in the joint order of `target_pkl`."""
    n_src, n_dst = _joint_names(source_pkl), _joint_names(target_pkl)
    moves = MOVES[convention]
    src = source_slots(n_src, n_dst, moves)
    with h5py.File(path, "r+") as f:
        md = f["metadata"].attrs
        assert int(md["n_joints"]) == len(n_src), f"the HDF5 has {md['n_joints']} joints, the source model {len(n_src)}"
        assert "joint_names" not in md or json.loads(md["joint_names"]) == n_src
        for name, ax in JOINT_DATASETS:
            d = f[name]
            a = d[()]
            assert a.shape[ax] == len(n_src), (name, a.shape)
            out = np.zeros(a.shape[:ax] + (len(n_dst),) + a.shape[ax + 1 :], a.dtype)
            idx = [slice(None)] * a.ndim
            for j, s in enumerate(src):
                if s >= 0:
                    idx_o, idx_s = list(idx), list(idx)
                    idx_o[ax], idx_s[ax] = j, s
                    out[tuple(idx_o)] = a[tuple(idx_s)]
            chunks, comp, opts = d.chunks, d.compression, d.compression_opts  # chunks is None for contiguous datasets
            del f[name]
            f.create_dataset(name, data=out, chunks=chunks, compression=comp, compression_opts=opts)
            print(f"[remap] {name}: {a.shape} -> {out.shape}", flush=True)
        md["n_joints"] = str(len(n_dst)) if isinstance(md["n_joints"], str) else len(n_dst)
        md["n_pose"] = str(len(n_dst) - 1) if isinstance(md["n_pose"], str) else len(n_dst) - 1
        md["joint_names"] = json.dumps(n_dst)
        md["proxy_remap"] = json.dumps(
            {
                "convention": convention,
                "moves": moves,
                "v5_pkl": source_pkl,
                "v6_pkl": target_pkl,
                "source_slot_per_joint": src.tolist(),
            }
        )
    print("[remap] done:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5", help="Multi-view HDF5 to modify IN PLACE (use a copy)")
    parser.add_argument("source_pkl", help="Model the HDF5 was built for (e.g. the 40-joint v5 mouse)")
    parser.add_argument("target_pkl", help="Model whose joint order to write (e.g. the 51-joint v6 mouse)")
    parser.add_argument(
        "convention", choices=sorted(MOVES), help="Labelling convention, i.e. which label moves to apply"
    )
    args = parser.parse_args()
    remap_joints_by_name(args.hdf5, args.source_pkl, args.target_pkl, args.convention)


if __name__ == "__main__":
    main()
