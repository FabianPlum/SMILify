"""PROBE 4: how much loss budget do the ant's thin structures actually get?

Assigns every template vertex/face to a body part via the dominant skinning weight,
then reports (a) share of surface AREA, (b) share of VERTEX COUNT. The gap between
the two is exactly the bias that `chamfer(target_samples, src.verts_padded())` injects.
Also converts area share into "points per leg segment" at the current 3000-sample budget.
"""

import pickle
import numpy as np

TEMPLATE = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
OUT = "/home/fabi/dev/SMILify/diagnostics/registration_part_budget_probe_out.txt"
N_SAMPLES = 3000  # what Stage.forward() uses today
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


d = pickle.load(open(TEMPLATE, "rb"), encoding="latin1")
v = np.asarray(d["v_template"], float)
f = np.asarray(d["f"], np.int64)
W = np.asarray(d["weights"], float)
names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["J_names"]]

dom = W.argmax(1)  # dominant joint per vertex
a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
face_part = dom[f].min(1)  # face -> a part (ties broken low)


def group(n):
    if n.startswith("l_"):
        return "legs"
    if n.startswith("an_"):
        return "antennae"
    if n.startswith("ma_"):
        return "mandibles"
    if n.startswith("b_a"):
        return "gaster/petiole"
    if n == "b_h":
        return "head"
    return "thorax/root"


g_v = np.array([group(names[i]) for i in dom])
g_f = np.array([group(names[i]) for i in face_part])

log(f"template: {len(v)} verts, {len(f)} faces, {len(names)} joints")
log("")
log(f"{'part group':<16} {'area %':>8} {'vert %':>8} {'ratio v/a':>10} {'pts@3000':>9}")
tot = area.sum()
for gname in ["thorax/root", "head", "gaster/petiole", "legs", "antennae", "mandibles"]:
    ap = area[g_f == gname].sum() / tot
    vp = (g_v == gname).mean()
    log(f"{gname:<16} {100 * ap:>7.1f}% {100 * vp:>7.1f}% {vp / max(ap, 1e-9):>10.2f} {ap * N_SAMPLES:>9.0f}")

log("")
n_leg_seg = sum(1 for n in names if n.startswith("l_"))
leg_area = area[g_f == "legs"].sum() / tot
log(f"leg joints in rig: {n_leg_seg}")
log(f"target points landing on ALL legs at {N_SAMPLES} samples: {leg_area * N_SAMPLES:.0f}")
log(f"  -> ~{leg_area * N_SAMPLES / max(n_leg_seg, 1):.1f} target points per leg segment")
ant_area = area[g_f == "antennae"].sum() / tot
n_ant = sum(1 for n in names if n.startswith("an_"))
log(
    f"target points landing on ALL antennae: {ant_area * N_SAMPLES:.0f} "
    f"(~{ant_area * N_SAMPLES / max(n_ant, 1):.1f} per antennal segment)"
)

# how much does raising the budget cost?
log("")
log("points per leg segment vs sample budget:")
for n in [3000, 10000, 30000, 100000]:
    log(f"  {n:>7} samples -> {leg_area * n / max(n_leg_seg, 1):>6.1f} per leg segment")
open(OUT, "w").write("\n".join(lines) + "\n")
print("written ->", OUT)
