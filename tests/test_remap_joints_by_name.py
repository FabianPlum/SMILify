"""remap_joints_by_name: move the per-joint arrays of a multi-view HDF5 to another model's joint order, by name."""

import json
import pickle

import h5py
import numpy as np
import pytest

from smal_fitter.multiview_common.remap_joints_by_name import JOINT_DATASETS, remap_joints_by_name

N, V = 3, 2
# Tree: Lumbar-Vertebrae -> (skull -> Ear_L), (tibia_L -> Foot_L), Tail_00, in depth-first order.
SRC = ["Lumbar-Vertebrae", "skull", "Ear_L", "tibia_L", "Foot_L", "Tail_00"]
# Target as the add-on exports it: each new leaf joint lands after its parent's subtree, so Tail_00 moves from 5 to 8.
DST = ["Lumbar-Vertebrae", "skull", "Ear_L", "head_falkner", "tibia_L", "Foot_L", "knee_L_falkner", "heel_L_falkner"]
DST += ["Tail_00", "hip_center_m103"]  # hip_center_m103 belongs to another convention: stays unlabelled
# Independent oracle for the "falkner" convention: target joint -> source joint whose labels it gets. skull, tibia_L
# and Foot_L hand their labels to their proxies and become unlabelled themselves.
FROM = {
    "Lumbar-Vertebrae": "Lumbar-Vertebrae",
    "Ear_L": "Ear_L",
    "head_falkner": "skull",
    "knee_L_falkner": "tibia_L",
    "heel_L_falkner": "Foot_L",
    "Tail_00": "Tail_00",
}


def _shape(name, axis, n_joints):
    trailing = {"keypoints_2d": (2,), "keypoints_2d_raw": (2,), "keypoints_3d": (3,), "joint_rot": (3,)}
    lead = (N, V) if axis == 2 else (N,)
    return lead + (n_joints,) + trailing.get(name.rsplit("/", 1)[1], ())


@pytest.mark.parametrize("layout", ["chunked", "contiguous"])
def test_remap_reorders_by_name_and_leaves_new_joints_unlabelled(tmp_path, layout):
    for tag, names in (("src", SRC), ("dst", DST)):
        with open(tmp_path / f"{tag}.pkl", "wb") as f:
            pickle.dump({"J_names": names}, f)
    rng = np.random.default_rng(0)
    data = {name: (1.0 + rng.random(_shape(name, ax, len(SRC)))).astype(np.float32) for name, ax in JOINT_DATASETS}
    K = rng.random((N, V, 3, 3)).astype(np.float32)
    path = tmp_path / "ds.h5"
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs.update(
            {"n_joints": len(SRC), "n_pose": len(SRC) - 1, "joint_names": json.dumps(SRC), "crop_mode": "centred"}
        )
        for name, a in data.items():
            kw = (
                {"chunks": (1,) + a.shape[1:], "compression": "gzip", "compression_opts": 4}
                if layout == "chunked"
                else {}
            )
            f.create_dataset(name, data=a, **kw)
        f.create_dataset("multiview_keypoints/camera_intrinsics", data=K)
        f.create_dataset("auxiliary/session_name", data=["s"] * N, dtype=h5py.string_dtype())

    remap_joints_by_name(str(path), str(tmp_path / "src.pkl"), str(tmp_path / "dst.pkl"), "falkner")

    with h5py.File(path, "r") as f:
        for name, ax in JOINT_DATASETS:
            d = f[name]
            out = d[()]
            assert out.shape == _shape(name, ax, len(DST)) and out.dtype == np.float32, name
            if layout == "chunked":  # the source's chunk shape and filter are kept
                assert (d.chunks, d.compression, d.compression_opts) == ((1,) + data[name].shape[1:], "gzip", 4), name
            else:
                assert d.chunks is None and d.compression is None, name
            for j, joint in enumerate(DST):
                got = np.take(out, j, axis=ax)
                if joint in FROM:
                    assert np.array_equal(got, np.take(data[name], SRC.index(FROM[joint]), axis=ax)), (name, joint)
                else:  # unlabelled: zero keypoints, visibility and confidence
                    assert not got.any(), (name, joint)
        np.testing.assert_array_equal(f["multiview_keypoints/camera_intrinsics"][()], K)
        assert [s.decode() for s in f["auxiliary/session_name"][()]] == ["s"] * N
        md = f["metadata"].attrs
        assert (int(md["n_joints"]), int(md["n_pose"]), md["crop_mode"]) == (len(DST), len(DST) - 1, "centred")
        assert json.loads(md["joint_names"]) == DST
        remap = json.loads(md["proxy_remap"])
        assert remap["convention"] == "falkner"
        assert remap["source_slot_per_joint"] == [0, -1, 2, 1, -1, -1, 3, 4, 5, -1]
