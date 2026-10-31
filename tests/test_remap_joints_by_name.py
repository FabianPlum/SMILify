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
# Independent oracle for the labels under the "falkner" convention: target joint -> source joint whose labels it gets.
# skull, tibia_L and Foot_L hand their labels to their proxies and become unlabelled themselves.
FROM = {
    "Lumbar-Vertebrae": "Lumbar-Vertebrae",
    "Ear_L": "Ear_L",
    "head_falkner": "skull",
    "knee_L_falkner": "tibia_L",
    "heel_L_falkner": "Foot_L",
    "Tail_00": "Tail_00",
}
BY_NAME = {n: n for n in DST if n in SRC}  # oracle for a plain reorder by name, and for the joint rotations
POSE = "parameters/joint_rot"


def _shape(name, axis, n_joints):
    trailing = {"keypoints_2d": (2,), "keypoints_2d_raw": (2,), "keypoints_3d": (3,), "joint_rot": (3,)}
    lead = (N, V) if axis == 2 else (N,)
    return lead + (n_joints,) + trailing.get(name.rsplit("/", 1)[1], ())


def _write_inputs(tmp_path, layout="chunked"):
    """src.pkl, dst.pkl and a source HDF5 whose per-joint datasets are all non-zero. Returns (path, data, K)."""
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
    return path, data, K


def _assert_joints_follow(out, source, ax, oracle, name):
    """Each target joint holds the data of the source joint `oracle` names, and every other joint is all zeros."""
    for j, joint in enumerate(DST):
        got = np.take(out, j, axis=ax)
        if joint in oracle:
            assert np.array_equal(got, np.take(source, SRC.index(oracle[joint]), axis=ax)), (name, joint)
        else:  # unlabelled (zero keypoints, visibility and confidence) or zero rotation
            assert not got.any(), (name, joint)


@pytest.mark.parametrize("layout", ["chunked", "contiguous"])
def test_remap_reorders_by_name_and_leaves_new_joints_unlabelled(tmp_path, layout):
    path, data, K = _write_inputs(tmp_path, layout)

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
            _assert_joints_follow(out, data[name], ax, BY_NAME if name == POSE else FROM, name)
        # Rotations are pose, not labels: skull keeps its own non-zero rotation and its proxy gets none.
        rot, rot_src = f[POSE][()], data[POSE]
        assert rot_src.all()
        np.testing.assert_array_equal(rot[:, DST.index("skull")], rot_src[:, SRC.index("skull")])
        assert not rot[:, DST.index("head_falkner")].any()
        np.testing.assert_array_equal(f["multiview_keypoints/camera_intrinsics"][()], K)
        assert [s.decode() for s in f["auxiliary/session_name"][()]] == ["s"] * N
        md = f["metadata"].attrs
        assert (int(md["n_joints"]), int(md["n_pose"]), md["crop_mode"]) == (len(DST), len(DST) - 1, "centred")
        assert json.loads(md["joint_names"]) == DST
        remap = json.loads(md["proxy_remap"])
        assert remap["convention"] == "falkner"
        assert remap["source_slot_per_joint"] == [0, -1, 2, 1, -1, -1, 3, 4, 5, -1]


def test_remap_refuses_a_target_without_the_proxy_joints(tmp_path):
    path, _, _ = _write_inputs(tmp_path)
    with open(tmp_path / "no_proxies.pkl", "wb") as f:  # e.g. the source model passed as the target by mistake
        pickle.dump({"J_names": SRC}, f)
    before = path.read_bytes()

    with pytest.raises(ValueError, match=r"\['head_falkner', 'heel_L_falkner', 'knee_L_falkner'\]"):
        remap_joints_by_name(str(path), str(tmp_path / "src.pkl"), str(tmp_path / "no_proxies.pkl"), "falkner")

    assert path.read_bytes() == before


def test_remap_convention_none_reorders_by_name(tmp_path):
    path, data, _ = _write_inputs(tmp_path)

    remap_joints_by_name(str(path), str(tmp_path / "src.pkl"), str(tmp_path / "dst.pkl"), "none")

    with h5py.File(path, "r") as f:
        for name, ax in JOINT_DATASETS:
            _assert_joints_follow(f[name][()], data[name], ax, BY_NAME, name)
        remap = json.loads(f["metadata"].attrs["proxy_remap"])
        assert (remap["moves"], remap["source_slot_per_joint"]) == ({}, [0, 1, 2, -1, 3, 4, -1, -1, 5, -1])


def test_remap_refuses_a_second_run_with_the_target_as_source(tmp_path):
    path, _, _ = _write_inputs(tmp_path)
    src, dst = str(tmp_path / "src.pkl"), str(tmp_path / "dst.pkl")
    remap_joints_by_name(str(path), src, dst, "falkner")
    before = path.read_bytes()

    # The file is in the target's joint order now, so only this refusal keeps a second falkner run from giving the
    # proxies the emptied slots of the moved joints.
    proxies = r"\['head_falkner', 'heel_L_falkner', 'knee_L_falkner'\]"
    with pytest.raises(ValueError, match=r"already has the proxy joints " + proxies):
        remap_joints_by_name(str(path), dst, dst, "falkner")

    assert path.read_bytes() == before


def test_remap_refuses_a_target_that_lacks_source_joints(tmp_path):
    path, _, _ = _write_inputs(tmp_path)
    src, dst = str(tmp_path / "src.pkl"), str(tmp_path / "dst.pkl")
    remap_joints_by_name(str(path), src, dst, "falkner")  # labelled proxies, as in a remapped v6 file
    before = path.read_bytes()

    # Back to the smaller model with no moves (v6 -> v5 with "none"): the proxies' labels would be deleted.
    dropped = r"\['head_falkner', 'knee_L_falkner', 'heel_L_falkner', 'hip_center_m103'\]"
    with pytest.raises(ValueError, match=r"lacks the source joints " + dropped):
        remap_joints_by_name(str(path), dst, src, "none")

    assert path.read_bytes() == before
