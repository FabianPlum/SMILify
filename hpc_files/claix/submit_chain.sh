#!/usr/bin/env bash
# Submit N chained links of a job script; each link starts when the previous one
# ends. afterany, not afterok: a link killed at the wall-clock limit is the normal
# case, and train.sbatch resumes from the newest checkpoint. A link that finds
# final_model.pth exits 0 at once, so over-provisioning the chain costs little.
#
#   CONFIG=configs_claix/run.json bash hpc_files/claix/submit_chain.sh 5 \
#       --account=<project> --export=ALL,CONFIG hpc_files/claix/train.sbatch
#
# Every argument after N is passed to each sbatch call unchanged.
# Links needed ~= ceil(remaining_epochs * hours_per_epoch / 23); time one link first.
# Cancel the whole chain: scancel <all printed job ids>
set -euo pipefail

n="${1:?usage: submit_chain.sh N <sbatch args...> <script.sbatch>}"
shift
prev=""
ids=()
for ((i = 1; i <= n; i++)); do
    jid="$(sbatch --parsable ${prev:+--dependency=afterany:$prev} "$@")"
    jid="${jid%%;*}" # --parsable prints "id;cluster" on multi-cluster setups
    echo "link $i/$n: $jid${prev:+ (after $prev)}"
    ids+=("$jid")
    prev="$jid"
done
echo "cancel all: scancel ${ids[*]}"
