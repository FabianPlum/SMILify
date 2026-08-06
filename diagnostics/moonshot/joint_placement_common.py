"""Shared helpers for the objective joint-placement probes (probe 20/21/22).

Everything here is a numpy re-implementation of the exact forward pass the fitter uses,
verified against the stored `verts` in each npz (see joint_placement_repro_PROBE.py).

Key structural facts READ OFF THE CODE (not assumed):
  * fitter_3d/trainer.py:224   smal_model(betas, theta, betas_logscale=..., betas_trans=...)
    is called WITHOUT `del_v`, so  v_shaped = v_template + betas @ shapedirs   only.
  * fitter_3d/trainer.py:240   `verts = verts + _deform_verts` happens AFTER skinning, i.e.
    deform_verts is a POSED-space offset and does NOT move the joints at all.
    => joint placement is a function of (betas, log_beta_scales, betas_trans, pose, trans)
       and of NOTHING else.
  * smal_model/batch_lbs.py:151 betas_trans is multiplied by [1, -1, 1] before use.
  * smal_model/batch_lbs.py:158 j_here = J[i] - J[parent] + trans_offset[i]; the offset is a
    LOCAL bone offset, applied before the parent's accumulated transform.
  * fitter_3d/trainer_moonshot.py:139  the template symmetry plane is y = 0.
"""

import numpy as np
import pickle


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    return dd


def rodrigues(theta):
    """theta (N,3) -> R (N,3,3). Matches smal_model/batch_lbs.batch_rodrigues."""
    theta = np.asarray(theta, dtype=np.float64)
    ang = np.linalg.norm(theta + 1e-8, axis=1, keepdims=True)
    r = theta / ang
    ang = ang[:, :, None]
    c = np.cos(ang)
    s = np.sin(ang)
    outer = r[:, :, None] * r[:, None, :]
    N = theta.shape[0]
    H = np.zeros((N, 3, 3))
    H[:, 0, 1] = -r[:, 2]
    H[:, 0, 2] = r[:, 1]
    H[:, 1, 0] = r[:, 2]
    H[:, 1, 2] = -r[:, 0]
    H[:, 2, 0] = -r[:, 1]
    H[:, 2, 1] = r[:, 0]
    eye = np.eye(3)[None]
    return c * eye + (1 - c) * outer + s * H


def global_rigid(Rs, J, parents, logscale=None, trans_off=None):
    """numpy port of batch_global_rigid_transformation (propagate_scaling=False).

    Rs        (N,J,3,3)
    J         (N,J,3)   joints before posing
    logscale  (N,J,3) or None
    trans_off (N,J,3) or None   -- RAW betas_trans, the [1,-1,1] flip is applied here
    returns   new_J (N,J,3), A (N,J,4,4)
    """
    N, nJ = J.shape[0], J.shape[1]
    S = np.exp(logscale) if logscale is not None else np.ones((N, nJ, 3))
    toff = None
    if trans_off is not None:
        toff = np.asarray(trans_off, dtype=np.float64) * np.array([1.0, -1.0, 1.0])

    res = np.zeros((N, nJ, 4, 4))
    res[:, 0, :3, :3] = Rs[:, 0]
    res[:, 0, :3, 3] = J[:, 0]
    res[:, 0, 3, 3] = 1.0

    for i in range(1, nJ):
        p = parents[i]
        j_here = J[:, i] - J[:, p]
        if toff is not None:
            j_here = j_here + toff[:, i]
        s_par_inv = 1.0 / S[:, p]  # diagonal inverse
        s_cur = S[:, i]
        # s_par_inv @ R @ s_cur  with both scales diagonal
        rot_new = s_par_inv[:, :, None] * Rs[:, i] * s_cur[:, None, :]
        A_here = np.zeros((N, 4, 4))
        A_here[:, :3, :3] = rot_new
        A_here[:, :3, 3] = j_here
        A_here[:, 3, 3] = 1.0
        res[:, i] = res[:, p] @ A_here

    new_J = res[:, :, :3, 3]
    Jw0 = np.concatenate([J, np.zeros((N, nJ, 1))], axis=2)[:, :, :, None]
    init_bone = res @ Jw0  # (N,J,4,1)
    A = res.copy()
    A[:, :, :, 3] -= init_bone[:, :, :, 0]
    return new_J, A


def joint_regress(v, J_reg):
    """v (N,V,3), J_reg (55,V) -> (N,55,3). Same contraction as smal_torch."""
    return np.einsum("jv,nvc->njc", J_reg, v)


def lr_pairs(J_names):
    """(i_right, i_left) index pairs, same rule as build_lr_joint_pairs."""
    idx = {n: i for i, n in enumerate(J_names)}
    out = []
    for n, i in idx.items():
        if n.endswith("_r"):
            twin = n[:-2] + "_l"
            if twin in idx:
                out.append((i, idx[twin]))
    return np.array(sorted(out))


MIRROR = np.array([1.0, -1.0, 1.0])  # template symmetry plane is y = 0


def joint_group(name):
    """Coarse anatomical group for a joint name."""
    if name.startswith("l_"):
        parts = name.split("_")
        seg = parts[2] if len(parts) > 3 else "?"
        if seg in ("co", "tr"):
            return "leg_proximal"
        if seg in ("fe", "ti"):
            return "leg_middle"
        if seg in ("ta", "pt"):
            return "leg_distal"
        return "leg_other"
    if name.startswith("a_"):
        return "antenna"
    if name.startswith("ma"):
        return "mandible"
    return "body"


# leg chain in proximal -> distal order; a "segment" is parent->child
LEG_SEGS = ["co", "tr", "fe", "ti", "ta", "pt"]


def leg_chain_indices(J_names):
    """{(leg, side): [idx_co, idx_tr, idx_fe, idx_ti, idx_ta, idx_pt]} where all exist."""
    idx = {n: i for i, n in enumerate(J_names)}
    out = {}
    for leg in ("1", "2", "3"):
        for side in ("l", "r"):
            chain = []
            ok = True
            for seg in LEG_SEGS:
                nm = f"l_{leg}_{seg}_{side}"
                if nm not in idx:
                    ok = False
                    break
                chain.append(idx[nm])
            if ok:
                out[(leg, side)] = chain
    return out


def align_template(v, I):
    """Port of smal_basics.align_smal_template_to_symmetry_axis (the ignore_sym=False path).

    Only the v_template transform matters here (the returned index sets are unused). Verified
    necessary: the ALL_ANTS_CLEAN fit reproduces to 4e-8 with it and only to 4e-3 without.
    """
    v = np.asarray(v, dtype=np.float64).copy()
    I = np.asarray(I).astype(int)
    v = v - np.mean(v)
    v[:, 1] = v[:, 1] - np.mean(v[I, 1])
    v[I, 1] = 0.0
    return v


def load_run(npz_path, model_path, align=False):
    d = np.load(npz_path, allow_pickle=True)
    dd = load_model(model_path)
    n = d["betas"].shape[0]
    nJ = dd["J_regressor"].shape[0]
    betas = d["betas"].astype(np.float64)
    nb = betas.shape[1]
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)  # (V,3,K)
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    if align:
        v_template = align_template(v_template, dd["sym_verts"])
    v_shaped = v_template[None] + np.einsum("nk,vck->nvc", betas, shapedirs[:, :, :nb])
    J_reg = np.asarray(dd["J_regressor"], dtype=np.float64)
    J = joint_regress(v_shaped, J_reg)
    theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], axis=1).astype(np.float64)
    Rs = rodrigues(theta.reshape(-1, 3)).reshape(n, nJ, 3, 3)
    lbs = d["log_beta_scales"].astype(np.float64)
    bt = d["betas_trans"].astype(np.float64) if "betas_trans" in d.files else None
    return dict(
        d=d,
        dd=dd,
        n=n,
        nJ=nJ,
        v_shaped=v_shaped,
        J=J,
        Rs=Rs,
        theta=theta,
        logscale=lbs,
        betas_trans=bt,
        trans=d["trans"].astype(np.float64),
        parents=np.asarray(dd["kintree_table"][0]).astype(int),
        J_names=list(dd["J_names"]),
        weights=np.asarray(dd["weights"], dtype=np.float64),
        faces=np.asarray(dd["f"]).astype(int),
        v_template=v_template,
    )


def winding_number(q, V, F, chunk=4000):
    """Generalized winding number (Jacobson et al. 2013) of points q wrt mesh (V,F).

    |w| > 0.5  <=>  inside, for a closed consistently-oriented mesh.
    q (Q,3), V (Nv,3), F (Nf,3). Returns (Q,).
    """
    q = np.asarray(q, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    out = np.zeros(q.shape[0])
    tri = V[F]  # (Nf,3,3)
    for s in range(0, F.shape[0], chunk):
        t = tri[s : s + chunk]
        a = t[None, :, 0, :] - q[:, None, :]
        b = t[None, :, 1, :] - q[:, None, :]
        c = t[None, :, 2, :] - q[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        num = np.einsum("qfi,qfi->qf", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("qfi,qfi->qf", a, b) * lc
            + np.einsum("qfi,qfi->qf", a, c) * lb
            + np.einsum("qfi,qfi->qf", b, c) * la
        )
        out += (2.0 * np.arctan2(num, den)).sum(axis=1)
    return out / (4.0 * np.pi)
