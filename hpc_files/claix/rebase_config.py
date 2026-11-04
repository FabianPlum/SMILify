#!/usr/bin/env python3
"""
Rewrite the absolute paths in a training config for another machine (e.g. CLAIX).

Run it on the cluster, from the repo root:

    python hpc_files/claix/rebase_config.py IN.json OUT.json \
        --map /home/fabi/dev/SMILify=/hpcwork/<project-id>/data \
        --map /media/fabi/Data/SMILify_RUNS=/hpcwork/<project-id>/runs

Every string value that starts with a mapped prefix is rewritten. The script then
exits non-zero if an input the run reads (dataset, model .pkl, resume/init
checkpoint) does not exist, so a wrong mapping fails on the login node instead of
after the queue wait. Absolute paths that no --map covers are listed as warnings.
Pass --no-check to prepare a config before its inputs are uploaded.

--set KEY=VALUE sets a dotted key; VALUE is parsed as JSON, else kept as a string.
Repeatable. Typical uses:
    --set output.save_checkpoint_every=2     (the trainer ignores --save_every_n_epochs
                                              for configs with a "mode" key, so set it here;
                                              a job killed at the wall loses every epoch
                                              since its last checkpoint)
    --set output.checkpoint_dir=/hpcwork/<project>/smoke/ckpt
                                             (smoke tests: train.sbatch resumes from the
                                              newest checkpoint in checkpoint_dir, so a smoke
                                              run must not share it with the real run)
"""

import argparse
import json
import os
import sys

# Keys whose values the trainer reads (everything else path-like is an output dir).
# "path" is the per-entry key of multi_dataset.datasets.
INPUT_KEYS = {"data_path", "path", "smal_file", "resume_checkpoint", "init_checkpoint"}


def rebase(node, maps, key, inputs, unmapped):
    if isinstance(node, dict):
        return {k: rebase(v, maps, k, inputs, unmapped) for k, v in node.items()}
    if isinstance(node, list):
        return [rebase(v, maps, key, inputs, unmapped) for v in node]
    if isinstance(node, str) and node:
        if node.startswith("/"):
            for old, new in maps:
                if node == old or node.startswith(old + "/"):
                    node = new + node[len(old) :]
                    break
            else:
                unmapped.append((key, node))
        if key in INPUT_KEYS:
            inputs.append((key, node))
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_in")
    parser.add_argument("config_out")
    parser.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="prefix mapping, repeatable")
    parser.add_argument("--no-check", action="store_true", help="skip the existence check of input files")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="set a dotted key, repeatable")
    args = parser.parse_args()

    maps = []
    for m in args.map:
        old, sep, new = m.partition("=")
        if not sep or not old.startswith("/") or not new:
            sys.exit(f"ERROR: --map expects /old/prefix=/new/prefix, got {m!r}")
        maps.append((old.rstrip("/"), new.rstrip("/")))

    with open(args.config_in) as f:
        cfg = json.load(f)
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            sys.exit(f"ERROR: --set expects dotted.key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    inputs, unmapped = [], []
    cfg = rebase(cfg, maps, None, inputs, unmapped)

    for key, path in unmapped if maps else []:
        print(f"WARNING: unmapped absolute path ({key}): {path}")
    missing = [(k, p) for k, p in inputs if not os.path.exists(p)]
    for key, path in inputs:
        print(f"  input {key:18s} {'MISSING' if (key, path) in missing else 'ok':7s} {path}")
    if missing and not args.no_check:
        sys.exit(f"ERROR: {len(missing)} input file(s) missing; fix the --map or upload them. Nothing written.")

    with open(args.config_out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {args.config_out}")


if __name__ == "__main__":
    main()
