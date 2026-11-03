#!/usr/bin/env python3
"""
Remap the per-joint arrays of a multi-view SLEAP HDF5 to another model's joint order, by joint name.

Reorders and extends every per-joint array of the HDF5 from the joint order of the source model (the model the HDF5
was built for) to the joint order of a target model, matching joints by name. A model that gains or reorders joints
needs this: the Blender add-on exports bones depth-first, so a new bone shifts the index of every joint after it.
Example: SMILy mouse v5 (40 joints) -> v6 (51 joints, v5 plus proxy leaf joints).

Writes a remapped copy (--output, a new file; the input is not modified) or rewrites the input (--in-place):

    python -m smal_fitter.multiview_common.remap_joints_by_name DATASET.h5 SOURCE.pkl TARGET.pkl CONVENTION \\
        --output REMAPPED.h5            # or: --in-place
        [--plan PROXY_BONE_PLAN.json]   # default: 3D_model_prep/mouse_limbfix/plans/proxy_bone_plan.json

- Labels (LABEL_DATASETS: 2D and 3D keypoints, visibility, confidence): a target joint that also exists in the source
  model keeps its labels, unless the convention moves them.
- CONVENTION selects one set of label moves from the proxy bone plan (--plan, by default the plan v6 was built from;
  see read_moves): each proxy joint of that convention takes over the labels of its label_slot joint, and that joint's
  own slot becomes unlabelled. A move whose joint the source model lacks is skipped. "none" moves nothing: a plain
  reorder by name.
- Every other target joint is unlabelled: all zeros (keypoints, visibility, confidence), which is how the HDF5 stores
  joints that were never labelled.
- A remap that would delete or overwrite labels is refused before the HDF5 is opened: a target model that lacks a
  proxy joint the convention moves labels to, a source model that already has one (a second run on a remapped file,
  for example), and source joints that the target model lacks and no move covers (v6 -> v5 with "none", for example).
  Source and target models with the same joints are refused too: there is nothing to remap, and a rerun would only
  overwrite the file's proxy_remap record.
- Joint rotations (POSE_DATASETS: parameters/joint_rot, root at index 0) are pose parameters, not labels. They follow
  the joint name only, whatever the convention: every joint keeps its own rotation, and a target joint that the
  source model lacks (a proxy joint, for example) gets zero rotation.
- Only the per-joint datasets (JOINT_DATASETS) are rewritten, keeping their dtype, chunking and compression. All of
  them are checked before the first one is rewritten (and before --output copies the file); only the raw-label ones
  (OPTIONAL_DATASETS) may be missing, as in merged and replicAnt HDF5s. Images, cameras and all other datasets are
  left untouched, so sidecars of per-sample data built on the source still pair with the copy. Per-joint sidecars do
  not: append the raw-label sidecar of add_raw_keypoints.py before the remap, not after (append checks the rows, not
  the joint count). The metadata gets n_joints, n_pose and joint_names in the target order, plus proxy_remap: the
  convention, the moves, both model paths ("source_pkl", "target_pkl") and the source slot of the labels of every
  target joint (-1 = unlabelled).
"""

import argparse
import json
import os
import pickle
import shutil

import h5py
import numpy as np

# The proxy bone plan the v6 mouse was built from: its proxies give the label moves of each convention.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROXY_PLAN = os.path.join(REPO_ROOT, "3D_model_prep", "mouse_limbfix", "plans", "proxy_bone_plan.json")
# Datasets with a joint axis: (name, axis). Labels follow the convention's moves.
LABEL_DATASETS = [
    ("multiview_keypoints/keypoints_2d", 2),
    ("multiview_keypoints/keypoints_2d_raw", 2),
    ("multiview_keypoints/keypoint_visibility", 2),
    ("multiview_keypoints/keypoint_visibility_raw", 2),
    ("multiview_keypoints/keypoint_confidence_raw", 2),
    ("multiview_keypoints/keypoints_3d", 1),
]
# Pose parameters follow the joint name only: joint_rot is (N, n_joints, 3) axis-angle, n_joints = n_pose + 1.
POSE_DATASETS = [("parameters/joint_rot", 1)]
JOINT_DATASETS = LABEL_DATASETS + POSE_DATASETS
# Raw SLEAP labels: only the SLEAP preprocessor and add_raw_keypoints.py write them, so merged and replicAnt HDF5s (and
# SLEAP HDF5s built before raw labels existed) have none. Remapped when present.
OPTIONAL_DATASETS = {name for name, _ in LABEL_DATASETS if name.endswith("_raw")}


def read_moves(plan_path=PROXY_PLAN):
    """Label moves per convention from a proxy bone plan: {convention: {label_slot joint: proxy joint}}, plus "none".

    Each proxy of the plan takes over the labels of its label_slot joint under its convention, in the plan's order. A
    plan may not use the convention "none", nor give one label_slot to two proxies of a convention.
    """
    with open(plan_path) as f:
        proxies = json.load(f)["proxies"]
    moves = {"none": {}}  # plain reorder by joint name
    # falkner: the Falkner lookup table maps the label Head to both skull and head_top. Only skull's copy moves (to
    # head_falkner), so head_top and head_falkner (0.056 u apart) share one label and per-label means count Head twice.
    for p in proxies:
        if p["convention"] == "none":  # its moves would turn the plain reorder into a move
            raise ValueError(f"{plan_path}: proxy {p['name']}: the convention 'none' is reserved for the plain reorder")
        conv = moves.setdefault(p["convention"], {})
        if p["label_slot"] in conv:  # a dict holds one proxy per label_slot: the second would be dropped
            raise ValueError(f"{plan_path}: {p['label_slot']} is the label_slot of two {p['convention']} proxies")
        conv[p["label_slot"]] = p["name"]
    return moves


def _joint_names(pkl_path):
    with open(pkl_path, "rb") as f:
        return list(pickle.load(f, encoding="latin1")["J_names"])


def source_slots(source_names, target_names, moves):
    """Source joint index for every target joint, matched by name (-1 = unlabelled).

    A move whose source joint is not in `source_names` is skipped, so its proxy joint gets -1.
    """
    src = []
    for n in target_names:
        moved_from = [k for k, v in moves.items() if v == n and k in source_names]
        if moved_from:
            src.append(source_names.index(moved_from[0]))
        elif n in source_names and n not in moves:
            src.append(source_names.index(n))
        else:
            src.append(-1)
    return np.array(src)


def remap_joints_by_name(path, source_pkl, target_pkl, convention, output=None, plan=PROXY_PLAN):
    """Rewrite the per-joint datasets of the HDF5 at `path` in the joint order of `target_pkl`.

    output: write the remapped HDF5 to this new file (refused if it exists) and leave `path` unchanged. Without it,
    `path` is modified IN PLACE. Every check runs before the file is copied or written.
    """
    all_moves = read_moves(plan)
    if convention not in all_moves:
        raise ValueError(f"convention {convention!r} is not in {plan}: choose one of {sorted(all_moves)}")
    moves = all_moves[convention]
    n_src, n_dst = _joint_names(source_pkl), _joint_names(target_pkl)
    # Refuse every remap that would delete or overwrite labels, before opening the file.
    missing = sorted(p for s, p in moves.items() if s in n_src and p not in n_dst)
    if missing:  # the moved labels would have no slot
        raise ValueError(f"{target_pkl} has no proxy joints {missing} for convention {convention!r}")
    present = sorted(p for p in moves.values() if p in n_src)
    if present:  # e.g. a second run on a remapped file: the proxies would get the emptied slots of the moved joints
        raise ValueError(f"{source_pkl} already has the proxy joints {present}: their labels would be overwritten")
    dropped = [n for n in n_src if n not in n_dst and n not in moves]
    if dropped:  # e.g. v6 -> v5 with "none"
        raise ValueError(f"{target_pkl} lacks the source joints {dropped}: their labels would be deleted")
    if n_src == n_dst:  # e.g. v6 -> v6 with "none": would only overwrite the file's proxy_remap record
        raise ValueError(f"{source_pkl} and {target_pkl} have the same joints: nothing to remap")
    label_src = source_slots(n_src, n_dst, moves)
    pose_src = source_slots(n_src, n_dst, {})  # no label moves: a joint keeps its rotation
    with h5py.File(path, "r") as f:  # check the file before anything is copied or rewritten
        md = f["metadata"].attrs
        if int(md["n_joints"]) != len(n_src):
            raise ValueError(f"the HDF5 has {md['n_joints']} joints, the source model {len(n_src)}")
        if "joint_names" in md and json.loads(md["joint_names"]) != n_src:
            raise ValueError(f"the HDF5's joint_names differ from the J_names of {source_pkl}")
        datasets = [(name, ax) for name, ax in JOINT_DATASETS if name in f or name not in OPTIONAL_DATASETS]
        for name, ax in datasets:
            if name not in f:
                raise ValueError(f"the HDF5 has no {name}")
            if f[name].shape[ax] != len(n_src):
                raise ValueError(f"{name} has shape {f[name].shape}, expected {len(n_src)} joints on axis {ax}")
    if output is not None and os.path.exists(output):
        raise FileExistsError(f"{output} exists; refusing to overwrite")
    try:
        if output is not None:
            print(f"[remap] copying {path} -> {output} ...", flush=True)
            shutil.copyfile(path, output)
            path = output
        with h5py.File(path, "r+") as f:
            md = f["metadata"].attrs
            for name, ax in datasets:
                src = pose_src if (name, ax) in POSE_DATASETS else label_src
                d = f[name]
                a = d[()]
                out = np.zeros(a.shape[:ax] + (len(n_dst),) + a.shape[ax + 1 :], a.dtype)
                idx = [slice(None)] * a.ndim
                for j, s in enumerate(src):
                    if s >= 0:
                        idx_o, idx_s = list(idx), list(idx)
                        idx_o[ax], idx_s[ax] = j, s
                        out[tuple(idx_o)] = a[tuple(idx_s)]
                chunks, comp, opts = d.chunks, d.compression, d.compression_opts  # chunks is None if contiguous
                del f[name]
                f.create_dataset(name, data=out, chunks=chunks, compression=comp, compression_opts=opts)
                print(f"[remap] {name}: {a.shape} -> {out.shape}", flush=True)
            md["n_joints"] = str(len(n_dst)) if isinstance(md["n_joints"], str) else len(n_dst)
            md["n_pose"] = str(len(n_dst) - 1) if isinstance(md.get("n_pose"), str) else len(n_dst) - 1
            md["joint_names"] = json.dumps(n_dst)
            md["proxy_remap"] = json.dumps(
                {
                    "convention": convention,
                    "moves": moves,
                    "source_pkl": source_pkl,
                    "target_pkl": target_pkl,
                    "source_slot_per_joint": label_src.tolist(),
                }
            )
    except BaseException:  # a half-written copy must not pass for a remapped file (the input is untouched)
        if output is not None and os.path.exists(output):
            os.remove(output)
        raise
    print("[remap] done:", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5", help="Multi-view HDF5 built for the source model")
    parser.add_argument("source_pkl", help="Model the HDF5 was built for (e.g. the 40-joint v5 mouse)")
    parser.add_argument("target_pkl", help="Model whose joint order to write (e.g. the 51-joint v6 mouse)")
    parser.add_argument(
        "convention",
        help='Labelling convention: a proxy convention of the plan (the shipped plan has falkner and m103), or "none" '
        "to reorder by name only",
    )
    parser.add_argument("--output", default=None, help="Write the remapped copy here (whole-file duplicate; new file)")
    parser.add_argument(
        "--in-place", action="store_true", help="Remap the input file directly (rewrites only the per-joint datasets)"
    )
    parser.add_argument(
        "--plan",
        default=PROXY_PLAN,
        help="Proxy bone plan whose proxies give the label moves (default: the shipped "
        "3D_model_prep/mouse_limbfix/plans/proxy_bone_plan.json)",
    )
    args = parser.parse_args(argv)
    if bool(args.output) == bool(args.in_place):
        parser.error("Choose exactly one of --output / --in-place")
    remap_joints_by_name(args.hdf5, args.source_pkl, args.target_pkl, args.convention, args.output, args.plan)


if __name__ == "__main__":
    main()
