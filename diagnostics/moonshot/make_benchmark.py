"""Build a fixed benchmark subset of half_workers for fast iteration.

Selection is deliberate, not random: we stratify by the roll misalignment measured in
probe 02/03 so that every intervention is tested on BOTH well-aligned specimens (where
the baseline should already do fine) and badly-rolled ones (where the baseline should
fail). A subset drawn only from easy specimens would hide exactly the failure we are
trying to fix; one drawn only from hard specimens would overstate gains.

Writes a directory of symlinks so the existing --mesh_dir interface works unchanged.
"""

import os
import sys
import json
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MESH_DIR = "/media/fabi/Data/SMILify_DATASETS_BACKUP/custom_processing/antscan_proofread_castes/half_workers"
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
BENCH_DIR = os.path.join(os.path.dirname(__file__), "bench12")
N_BENCH = 12


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Reuse probe 03's measurements if present, else recompute on a wide sample.
    cache = os.path.join(OUT_DIR, "probe03_roll_validity.json")
    if os.path.exists(cache):
        with open(cache) as f:
            rows = json.load(f)["rows"]
        print(f"Using cached roll measurements for {len(rows)} specimens")
    else:
        raise SystemExit("run probe_03_roll_validity.py first")

    rows = sorted(rows, key=lambda r: r["roll_deg"])
    n = len(rows)

    # stratified pick: 4 low-roll, 4 mid, 4 high
    thirds = [rows[: n // 3], rows[n // 3 : 2 * n // 3], rows[2 * n // 3 :]]
    picked = []
    for band, name in zip(thirds, ["low", "mid", "high"]):
        idx = np.linspace(0, len(band) - 1, N_BENCH // 3).astype(int)
        for i in idx:
            r = dict(band[i])
            r["band"] = name
            picked.append(r)

    # de-dup by genus so we do not spend the whole benchmark on one genus
    seen_genus = {}
    final = []
    for r in picked:
        genus = r["file"].split("_")[0]
        if seen_genus.get(genus, 0) >= 2:
            continue
        seen_genus[genus] = seen_genus.get(genus, 0) + 1
        final.append(r)

    if os.path.isdir(BENCH_DIR):
        shutil.rmtree(BENCH_DIR)
    os.makedirs(BENCH_DIR)

    print(f"\nBenchmark set ({len(final)} specimens):")
    print(f"{'band':>5}  {'roll':>6}  {'e3/e2':>6}  file")
    for r in final:
        src = os.path.join(MESH_DIR, r["file"])
        dst = os.path.join(BENCH_DIR, r["file"])
        os.symlink(src, dst)
        print(f"{r['band']:>5}  {r['roll_deg']:6.1f}  {r['degeneracy_e3_over_e2']:6.3f}  {r['file']}")

    with open(os.path.join(OUT_DIR, "benchmark_set.json"), "w") as f:
        json.dump(final, f, indent=1)
    print(f"\nSymlinked into {BENCH_DIR}")
    print(f"Manifest: {os.path.join(OUT_DIR, 'benchmark_set.json')}")


if __name__ == "__main__":
    main()
