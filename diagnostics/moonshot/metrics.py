"""Evaluation suite for ant template registration.

There is no ground-truth correspondence for these scans, so the suite deliberately mixes
three kinds of measurement:

  A. SURFACE AGREEMENT (does the fit reach the target?)
       chamfer, F-score@tau, 95th-pct Hausdorff, normal consistency
     These are what the optimizer already maximizes, so they are necessary but NOT
     sufficient -- free-form deform_verts can drive them down while destroying anatomy.

  B. MESH / DEFORMATION VALIDITY (is the fit a plausible ant, or a melted one?)
       edge-length distortion, triangle-quality collapse, normal flips,
       ARAP-style local rigidity, deform_verts magnitude
     These catch the failure mode that (A) is blind to.

  C. CORRESPONDENCE PROXIES WITHOUT GROUND TRUTH (is the map anatomically sane?)
       bilateral symmetry consistency  -- the template ships a sym_verts map, and every
           target is bilaterally symmetric, so a correct registration must map
           left-right mirrored template vertices to left-right mirrored target locations.
           This is FREE ground truth and the strongest signal available here.
       per-part target coverage -- segment the template by skinning weights into
           body/legs/head/antennae and ask whether each part actually reaches target
           surface. Catches 'legs never left the rest pose', which chamfer hides because
           legs are a small fraction of surface area.
       part-wise chamfer -- same, but per anatomical group, so leg failure cannot be
           averaged away by a good torso fit.

Design note on tau for F-score: all meshes are normalized to max|coord| = 1, so a length
of 0.01 is 1% of the half-extent of the specimen. We report F-score at several tau rather
than committing to one.
"""

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points

# ---------------------------------------------------------------- helpers


def _as_meshes(verts, faces, device):
    if isinstance(verts, torch.Tensor) and verts.dim() == 2:
        verts = [verts]
        faces = [faces]
    return Meshes(verts=[v.to(device) for v in verts], faces=[f.to(device) for f in faces])


def _pts_and_normals(mesh, n):
    p, nrm = sample_points_from_meshes(mesh, n, return_normals=True)
    return p, nrm


# ---------------------------------------------------------------- A. surface agreement


def surface_metrics(pred_mesh, tgt_mesh, n_points=30000, taus=(0.005, 0.01, 0.02, 0.05)):
    """Bidirectional surface agreement between a predicted mesh and a target mesh.

    Samples points by AREA (not vertices), which matters here because the target scans
    are ~48k verts with wildly uneven tessellation while the template is ~10k. Comparing
    raw vertex sets would weight densely-tessellated regions far more heavily -- which is
    exactly what the live pipeline does (it chamfers 3000 sampled target points against
    ALL template verts).
    """
    with torch.no_grad():
        pp, pn = _pts_and_normals(pred_mesh, n_points)
        tp, tn = _pts_and_normals(tgt_mesh, n_points)

        # pred -> target and target -> pred nearest neighbours
        d_pt = knn_points(pp, tp, K=1)
        d_tp = knn_points(tp, pp, K=1)
        dist_pt = d_pt.dists[..., 0].sqrt()  # (1,N) euclidean
        dist_tp = d_tp.dists[..., 0].sqrt()

        out = {}
        out["chamfer_l2"] = float((dist_pt.pow(2).mean() + dist_tp.pow(2).mean()).item())
        out["chamfer_l1"] = float((dist_pt.mean() + dist_tp.mean()).item())
        out["acc_mean"] = float(dist_pt.mean().item())  # pred->tgt: accuracy
        out["comp_mean"] = float(dist_tp.mean().item())  # tgt->pred: completeness
        out["hausdorff_95"] = float(max(torch.quantile(dist_pt, 0.95).item(), torch.quantile(dist_tp, 0.95).item()))
        out["hausdorff_max"] = float(max(dist_pt.max().item(), dist_tp.max().item()))

        for tau in taus:
            prec = (dist_pt < tau).float().mean()
            rec = (dist_tp < tau).float().mean()
            f = 2 * prec * rec / (prec + rec + 1e-12)
            out[f"fscore@{tau}"] = float(f.item())
            out[f"precision@{tau}"] = float(prec.item())
            out[f"recall@{tau}"] = float(rec.item())

        # normal consistency along the correspondence
        nn_tn = torch.gather(tn, 1, d_pt.idx[..., 0:1].expand(-1, -1, 3))
        cos = torch.nn.functional.cosine_similarity(pn, nn_tn, dim=-1).abs()
        out["normal_consistency"] = float(cos.mean().item())
        out["normal_flip_frac"] = float(
            (torch.nn.functional.cosine_similarity(pn, nn_tn, dim=-1) < 0).float().mean().item()
        )
    return out


# ---------------------------------------------------------------- B. deformation validity


def deformation_metrics(pred_verts, template_verts, faces, deform_verts=None):
    """How badly has the template been mangled to achieve its fit?

    All quantities are computed against the TEMPLATE's own rest geometry, so they measure
    distortion introduced by the fit rather than anything about the target.
    """
    with torch.no_grad():
        out = {}
        v0 = template_verts
        v1 = pred_verts
        f = faces

        # per-edge length ratio (unique edges from faces)
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
        l0 = (v0[e[:, 0]] - v0[e[:, 1]]).norm(dim=-1)
        l1 = (v1[e[:, 0]] - v1[e[:, 1]]).norm(dim=-1)
        ratio = l1 / (l0 + 1e-12)
        out["edge_ratio_mean"] = float(ratio.mean().item())
        out["edge_ratio_std"] = float(ratio.std().item())
        out["edge_stretch_p95"] = float(torch.quantile(ratio, 0.95).item())
        out["edge_compress_p05"] = float(torch.quantile(ratio, 0.05).item())
        # log-ratio is the symmetric measure of distortion
        out["edge_logratio_absmean"] = float(ratio.clamp_min(1e-9).log().abs().mean().item())

        # triangle area ratio + degenerate triangles
        def tri_area(v):
            t = v[f]
            return 0.5 * torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1).norm(dim=-1)

        a0, a1 = tri_area(v0), tri_area(v1)
        ar = a1 / (a0 + 1e-12)
        out["area_ratio_mean"] = float(ar.mean().item())
        out["area_logratio_absmean"] = float(ar.clamp_min(1e-9).log().abs().mean().item())
        out["degenerate_tri_frac"] = float((a1 < 1e-8).float().mean().item())

        # triangle quality (radius ratio proxy): 4*sqrt(3)*A / sum(edge^2), 1 = equilateral
        t = v1[f]
        e0 = (t[:, 1] - t[:, 0]).norm(dim=-1)
        e1 = (t[:, 2] - t[:, 1]).norm(dim=-1)
        e2 = (t[:, 0] - t[:, 2]).norm(dim=-1)
        q = 4 * np.sqrt(3) * a1 / (e0**2 + e1**2 + e2**2 + 1e-12)
        out["tri_quality_mean"] = float(q.mean().item())
        out["tri_quality_p05"] = float(torch.quantile(q, 0.05).item())

        if deform_verts is not None:
            dn = deform_verts.norm(dim=-1)
            out["deform_mag_mean"] = float(dn.mean().item())
            out["deform_mag_p95"] = float(torch.quantile(dn, 0.95).item())
            out["deform_mag_max"] = float(dn.max().item())

        # SURFACE COHERENCE -- added after the suite missed visibly torn meshes.
        # Everything above is local and scale-free, so a mesh can tear into well-shaped,
        # well-proportioned pieces and still score well. The angle between adjacent face
        # normals catches that; the p99 is the discriminating statistic, because a smoothly
        # deformed mesh keeps a low tail while one ripped in a few places does not.
        # Measured M6_augmented at p99 = 142.7 deg / 2.79% folded, vs baseline 70.7 / 0.54%.
        e2f = _face_adjacency(f)
        if e2f is not None:
            a, b = e2f
            t = v1[f]
            n = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
            n = torch.nn.functional.normalize(n, dim=-1)
            ang = torch.rad2deg(torch.arccos((n[a] * n[b]).sum(-1).clamp(-1, 1)))
            out["dihedral_mean"] = float(ang.mean().item())
            out["dihedral_p99"] = float(torch.quantile(ang, 0.99).item())
            out["folded_face_frac"] = float((ang > 90).float().mean().item())
    return out


_ADJ_CACHE = {}


def _face_adjacency(faces):
    """Pairs of face indices sharing an edge. Cached: it depends only on the template
    topology, which never changes across specimens or arms."""
    key = (faces.shape[0], int(faces[0, 0]), int(faces[-1, -1]))
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]
    try:
        e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
        e, _ = torch.sort(e, dim=1)
        fid = torch.arange(faces.shape[0], device=faces.device).repeat(3)
        k = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
        order = torch.argsort(k)
        k, fid = k[order], fid[order]
        idx = torch.nonzero(k[1:] == k[:-1], as_tuple=True)[0]
        res = (fid[idx], fid[idx + 1])
    except Exception:
        res = None
    _ADJ_CACHE[key] = res
    return res


# ---------------------------------------------------------------- C. correspondence proxies


def symmetry_metrics(pred_verts, sym_verts, template_verts):
    """Bilateral symmetry consistency -- the closest thing to free ground truth here.

    sym_verts (shipped in SMIL_OmniAnt.pkl) lists template vertices that lie on the
    midsagittal plane, i.e. vertices that SHOULD stay at y ~ 0 under a correct fit to a
    bilaterally-symmetric target. If a fit drags them off the midline, it has broken the
    anatomy -- typically because a leg or antenna got matched to the wrong side.

    Returns midline deviation in absolute units and normalized by specimen size.
    """
    with torch.no_grad():
        out = {}
        idx = sym_verts.long()
        y_pred = pred_verts[idx, 1]
        y_tmpl = template_verts[idx, 1]
        scale = pred_verts.abs().max().clamp_min(1e-9)
        out["midline_dev_mean"] = float(y_pred.abs().mean().item())
        out["midline_dev_p95"] = float(torch.quantile(y_pred.abs(), 0.95).item())
        out["midline_dev_mean_norm"] = float((y_pred.abs().mean() / scale).item())
        # how much WORSE than the template's own (should be ~0) midline
        out["midline_dev_excess"] = float((y_pred.abs().mean() - y_tmpl.abs().mean()).item())
    return out


def part_metrics(pred_mesh, tgt_mesh, part_vertex_ids, part_names, n_points=30000, tau=0.02):
    """Per-anatomical-part surface agreement.

    Legs and antennae are a small fraction of an ant's surface area, so a global chamfer
    can look fine while every leg is unmatched. This splits the template by skinning-weight
    assignment and asks, per part: how far is this part from the target surface, and does
    it actually reach it.

    Note the asymmetry: we measure part -> target only. Target -> part is not defined
    without a target segmentation, which we do not have.
    """
    with torch.no_grad():
        out = {}
        tp = sample_points_from_meshes(tgt_mesh, n_points)
        pv = pred_mesh.verts_packed()
        for name, ids in zip(part_names, part_vertex_ids):
            if len(ids) == 0:
                continue
            sub = pv[ids.long()].unsqueeze(0)
            d = knn_points(sub, tp, K=1).dists[..., 0].sqrt()
            out[f"part_{name}_dist_mean"] = float(d.mean().item())
            out[f"part_{name}_dist_p95"] = float(torch.quantile(d, 0.95).item())
            out[f"part_{name}_within_tau"] = float((d < tau).float().mean().item())
    return out


def template_part_segmentation(weights, joint_names):
    """Segment template vertices into anatomical groups by dominant skinning weight.

    weights: (V, J) linear blend skinning weights from the model pkl
    joint_names: list of J names, e.g. 'l_1_fe_r', 'b_h', 'an_2_l'
    """
    dom = np.asarray(weights).argmax(axis=1)
    groups = {"body": [], "head": [], "antenna": [], "mandible": [], "leg": [], "leg_distal": []}
    for j, nm in enumerate(joint_names):
        if nm.startswith("b_a") or nm == "b_t":
            g = "body"
        elif nm == "b_h":
            g = "head"
        elif nm.startswith("an_"):
            g = "antenna"
        elif nm.startswith("ma_"):
            g = "mandible"
        elif nm.startswith("l_"):
            # distal = tibia, tarsus, pretarsus -- the parts that fail first
            g = "leg_distal" if any(s in nm for s in ["_ti_", "_ta_", "_pt_"]) else "leg"
        elif nm.startswith("w_"):
            g = "body"
        else:
            g = "body"
        groups[g].append(j)

    part_ids, part_names = [], []
    for g, js in groups.items():
        if not js:
            continue
        mask = np.isin(dom, js)
        part_ids.append(torch.tensor(np.nonzero(mask)[0]))
        part_names.append(g)
    return part_ids, part_names


# ---------------------------------------------------------------- top level


def evaluate(
    pred_verts,
    faces,
    tgt_mesh,
    template_verts,
    sym_verts,
    part_ids,
    part_names,
    deform_verts=None,
    device="cuda:0",
    n_points=30000,
):
    """Run the full suite for ONE specimen. Returns a flat dict."""
    pred_mesh = _as_meshes(pred_verts, faces, device)
    r = {}
    r.update(surface_metrics(pred_mesh, tgt_mesh, n_points=n_points))
    r.update(deformation_metrics(pred_verts.to(device), template_verts.to(device), faces.to(device), deform_verts))
    r.update(symmetry_metrics(pred_verts.to(device), sym_verts.to(device), template_verts.to(device)))
    r.update(part_metrics(pred_mesh, tgt_mesh, part_ids, part_names, n_points=n_points))
    return r
