"""Generate single-variable ablation configs on the BASELINE's exact schedule.

The first sweep (E1/E3) changed several things at once AND shortened the deform budget
(850 iterations vs the baseline's 2000), so its arms were not attributable. This file
fixes that: every arm below uses the baseline's exact stage structure, iteration counts,
learning rates and loss weights, and changes exactly ONE thing.

Arms
  C0_control   baseline schedule inside the moonshot harness. NOT identical to the stock
               pipeline: it area-samples BOTH meshes (the stock loss compares 3000 area
               samples against all 10,229 raw template vertices) and uses a cosine LR
               decay. C0 vs runs/baseline therefore MEASURES those two changes, and C0 is
               the correct control for every other arm.
  A1_offset    + w_offset on the deform stages. Tests: can the fit keep its surface
               accuracy if free-form vertex offsets are made expensive?
  A2_restedge  edge_mode 'rest' instead of 'shrink'. Tests the mesh_edge_loss finding
               (target_length=0 penalises edge length itself => shrinkage).
  A3_priors    + w_beta_prior + w_sym. Tests the two priors that ship with the model and
               are never used (betas_prec is dead code; nothing ties L/R joint scales).
  A4_nofreeze  deform stages use scheme 'all' instead of 'deform', so pose is never
               frozen. Tests the headline structural finding directly.
  A5_robust    + Geman-McClure at a scale matched to the OBSERVED residual, not far below
               it. E3 used scale 0.02-0.05 while typical residuals are ~0.04, which put
               ~83% of points in the kernel's saturated zero-gradient region and starved
               the optimizer. Baseline Stage_1 chamfer_l2 = 0.0032 => RMS residual ~0.04,
               so the scale must be ~0.1-0.2 to reject only genuine debris.
"""

import copy
import os

import yaml

OUT = os.path.join(os.path.dirname(__file__), "cfg")
RUNS = "diagnostics/moonshot/runs"

# ---- the baseline schedule, transcribed exactly from fitter_3d/ants_cfg.yaml ----
BASE = {
    "Stage_0_init": dict(scheme="init_rot_lock", nits=100, lr=0.05),
    "Stage_1_default": dict(
        scheme="default",
        nits=300,
        lr=0.02,
        loss_weights={"w_chamfer": 1.0, "w_edge": 0.8, "w_normal": 0.02, "w_laplacian": 0.01},
        custom_lrs={"joint_rot": 0.002},
    ),
    "Stage_2_deform_coarse": dict(
        scheme="deform",
        nits=1000,
        lr=0.002,
        loss_weights={"w_chamfer": 1.0, "w_edge": 0.8, "w_normal": 0.005, "w_laplacian": 0.01},
    ),
    "Stage_3_deform_fine": dict(
        scheme="deform",
        nits=1000,
        lr=0.0005,
        loss_weights={"w_chamfer": 0.5, "w_edge": 0.2, "w_normal": 0.002, "w_laplacian": 0.001},
    ),
}

# harness settings applied to every stage of every arm, so they never differ between arms
COMMON = dict(edge_mode="shrink", n_sample=6000, lr_decay=False, robust_kernel="l2")


def make(name, mutate=None):
    stages = copy.deepcopy(BASE)
    for st in stages.values():
        st.update(copy.deepcopy(COMMON))
    if mutate:
        mutate(stages)
    cfg = {"stages": stages, "args": {"results_dir": f"{RUNS}/{name}"}}
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def m_offset(s):
    # deform stages only -- that is where the offsets are actually produced
    s["Stage_2_deform_coarse"]["loss_weights"]["w_offset"] = 5.0
    s["Stage_3_deform_fine"]["loss_weights"]["w_offset"] = 2.0


def m_restedge(s):
    for st in s.values():
        st["edge_mode"] = "rest"
    # The two edge losses have different natural magnitudes: the shrink form is
    # mean(edge_len^2) ~ 1e-4 while the rest form is mean((l/l0 - 1)^2), which is 0 at the
    # start and O(1e-2) once deformed. Keeping w_edge identical would silently change the
    # regularization strength as well as its form, so scale the weight to put the two
    # terms in a comparable range and keep this a test of FORM, not strength.
    for st in s.values():
        if "loss_weights" in st and "w_edge" in st["loss_weights"]:
            st["loss_weights"]["w_edge"] = round(st["loss_weights"]["w_edge"] * 0.02, 5)


def m_priors(s):
    for k, st in s.items():
        lw = st.setdefault("loss_weights", {})
        lw["w_sym"] = 0.5
        if st["scheme"] in ("default", "pose", "all"):
            lw["w_beta_prior"] = 0.002


def m_nofreeze(s):
    # keep iteration counts and lrs identical; only the parameter group changes, and
    # joint_rot keeps the same lr it had in Stage_1 so this is not a pose-lr experiment
    for k in ("Stage_2_deform_coarse", "Stage_3_deform_fine"):
        s[k]["scheme"] = "all"
        s[k]["custom_lrs"] = {"joint_rot": 0.002}


def m_robust(s):
    for st in s.values():
        st["robust_kernel"] = "gm"
        st["robust_scale"] = 0.20
        st["robust_scale_end"] = 0.10
        st["trim_frac"] = 0.02


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    arms = [
        ("C0_control", None),
        ("A1_offset", m_offset),
        ("A2_restedge", m_restedge),
        ("A3_priors", m_priors),
        ("A4_nofreeze", m_nofreeze),
        ("A5_robust", m_robust),
    ]
    for name, fn in arms:
        print("wrote", make(name, fn))
