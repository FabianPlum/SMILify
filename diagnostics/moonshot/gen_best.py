"""Generate 'best combination' arms from what the ablations actually showed.

Ablation results that drive this (all vs C0_control, 50 paired specimens, sign test):
  A4_nofreeze  scheme 'deform' -> 'all' in the late stages. Improved EVERY metric at once:
               fscore@0.01 +0.7% (35/50, p=0.007), normal_consistency +1.1% (40/50),
               edge_logratio -2.7% (39/50), tri_quality +1.4% (40/50), and
               deform_mag_mean -23.5% / p95 -22.0% (50/50, p=1.8e-15). Zero extra cost.
               => take unconditionally.
  A3_priors    the two dead priors. Neutral on final surface, but part_leg_distal
               -20.2% at the pose checkpoint (41/50, p=5.6e-6) and edge_logratio -8.3%
               at the final stage (45/50). => take, cheap.
  A1_offset    w_offset=5.0 cut deform_mag 89% and edge_logratio 66%, but cost
               fscore@0.01 -43% and part_leg_distal +371%. Far too strong.
               => sweep a much gentler weight instead of discarding the idea. This mirrors
                  Khaoula's penetration-weight finding, where 0.1/0.2 -> 0.02/0.05 kept 76%
                  of the benefit for 27.5% of the cost.
  A2_restedge  helped integrity and legs at the pose checkpoint but cost a lot of surface
               accuracy. => leave out of the headline combination, keep as its own arm.
  A5_robust    fscore@0.01 +13.1% at the pose checkpoint (48/50, p=2e-12) but
               part_leg_distal +230% at the final stage, which trips the pre-registered
               per-part regression gate. => leave out; revisit with a per-part scale.
"""

import copy
import os

import yaml

CFG = os.path.dirname(os.path.abspath(__file__)) + "/cfg"


def main():
    base = yaml.safe_load(open(os.path.join(CFG, "A4_nofreeze.yaml")))
    for w in [0.2, 1.0]:
        c = copy.deepcopy(base)
        for k in ("Stage_2_deform_coarse", "Stage_3_deform_fine"):
            c["stages"][k].setdefault("loss_weights", {})["w_offset"] = w if "coarse" in k else w * 0.4
        for k, st in c["stages"].items():
            st.setdefault("loss_weights", {})["w_sym"] = 0.5
            if st["scheme"] in ("default", "pose", "all"):
                st["loss_weights"]["w_beta_prior"] = 0.002
        name = f"B1_best_off{str(w).replace('.', 'p')}"
        c["args"]["results_dir"] = f"diagnostics/moonshot/runs/{name}"
        with open(os.path.join(CFG, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print("wrote", name)


if __name__ == "__main__":
    main()
