#!/usr/bin/env bash
#
# Purge third-party SMPL/SMAL blobs from the ENTIRE git history (issue #84).
#
# Untracking these files at HEAD (done in the accompanying commit) stops them
# being distributed going forward, but every historical commit still carries the
# blobs and they stay downloadable. Removing them for real means rewriting all
# history and force-pushing.
#
#   ./scripts/purge_licensed_blobs.sh            # dry run: report only, no writes
#   ./scripts/purge_licensed_blobs.sh --execute  # rewrite a mirror clone locally
#
# Even --execute does NOT touch the remote. It rewrites a throwaway mirror clone
# under /tmp and prints the force-push commands for you to run deliberately.
#
# READ THIS FIRST -- a rewrite is NOT sufficient on its own:
#   1. Force-pushing rewrites all 14 branches. Open PRs (#79, #81) will need
#      rebasing; every existing clone must re-clone. Coordinate first.
#   2. GitHub keeps unreachable blobs reachable by SHA and serves them from the
#      5 forks. After force-pushing you MUST open a GitHub Support ticket asking
#      them to purge cached views, and ask fork owners to delete/re-fork.
#   3. Treat the SMPL/SMAL copies as having been public. If that matters
#      contractually, tell MPI rather than relying on a silent rewrite.
#
set -euo pipefail

# Source to clone. Override to rehearse the rewrite against a local copy
# without hitting the network, e.g.  PURGE_SOURCE=. ./scripts/purge_licensed_blobs.sh --execute
PURGE_SOURCE="${PURGE_SOURCE:-https://github.com/FabianPlum/SMILify}"

# Deliberately NOT /tmp: the rewritten mirror must survive between the rewrite
# and the force-push, and /tmp is wiped on WSL session restart (observed during
# testing -- the mirror disappeared between two commands). Override with
# PURGE_WORKDIR if you want it elsewhere.
WORKDIR="${PURGE_WORKDIR:-${HOME}/smilify-purge}"
MIRROR="${WORKDIR}/SMILify.git"

# Paths to erase from every commit. The whole smal_model/smpl/ tree goes: it
# held only the two MPI SMPL models, an MPI-copyrighted __init__ stub and a
# .DS_Store, and nothing imports it.
PURGE_PATHS=(
  "smal_model/smpl/"
  "data/priors/walking_toy_symmetric_pose_prior_with_cov_35parts.pkl"
  "data/priors/walking_toy_symmetric_pose_prior_with_cov_35parts_WIN.pkl"
  "data/priors/zebra_walking_symmetric_pose_prior_with_cov_35parts.pkl"
)

# NOT purged -- our own models, which only borrow SMPL's file naming from early
# development. Listed here so nobody "helpfully" adds them later:
#   3D_model_prep/SMPL_fit*.pkl  SMPL_STICK.pkl  smpl_ATTA.pkl
#   3D_model_prep/SMIL_*.pkl  SMILy_*.pkl
#   data/priors/unity_*

EXECUTE=0
[[ "${1:-}" == "--execute" ]] && EXECUTE=1

echo "=== SMILify licensed-blob purge (issue #84) ==="
echo
echo "Paths to erase from all history:"
printf '  %s\n' "${PURGE_PATHS[@]}"
echo

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is not installed. Install it with:"
  echo "    pip install git-filter-repo"
  echo "(or: apt install git-filter-repo)"
  [[ $EXECUTE -eq 1 ]] && exit 1
  echo
fi

echo "--- Current footprint in history ---"
for p in "${PURGE_PATHS[@]}"; do
  n=$(git log --all --oneline -- "$p" 2>/dev/null | wc -l | tr -d ' ')
  echo "  ${n} commit(s) touch  ${p}"
done
echo
echo "  total commits in repo : $(git rev-list --all --count)"
echo "  branches on origin    : $(git ls-remote --heads origin 2>/dev/null | wc -l | tr -d ' ')"
echo "  .git size             : $(du -sh .git 2>/dev/null | cut -f1)"
echo

if [[ $EXECUTE -eq 0 ]]; then
  cat <<'EOF'
--- DRY RUN. Nothing was modified. ---

To rewrite a local mirror clone (still no remote writes):
    ./scripts/purge_licensed_blobs.sh --execute
EOF
  exit 0
fi

echo "--- Rewriting a mirror clone (remote untouched) ---"
echo "    source: ${PURGE_SOURCE}  (read-only; the rewrite happens in the clone)"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
# --no-hardlinks matters when PURGE_SOURCE is a local path: without it git
# hardlinks object files into the clone, so the rehearsal would share object
# storage with the real repo. We want the copy fully independent.
git clone --mirror --no-hardlinks "$PURGE_SOURCE" "$MIRROR"

filter_args=()
for p in "${PURGE_PATHS[@]}"; do filter_args+=(--path "$p"); done

git -C "$MIRROR" filter-repo --invert-paths "${filter_args[@]}" --force

echo
echo "--- Verifying the blobs are gone from the rewritten mirror ---"
fail=0
for p in "${PURGE_PATHS[@]}"; do
  if git -C "$MIRROR" log --all --oneline -- "$p" 2>/dev/null | grep -q .; then
    echo "  STILL PRESENT: $p"; fail=1
  else
    echo "  purged: $p"
  fi
done

echo
echo "--- Confirming our own models survived ---"
for p in 3D_model_prep/SMPL_fit_new.pkl 3D_model_prep/smpl_ATTA.pkl \
         3D_model_prep/SMPL_STICK.pkl 3D_model_prep/SMILy_STICK.pkl \
         data/priors/unity_pose_prior_with_cov_35parts.pkl; do
  if git -C "$MIRROR" cat-file -e "HEAD:$p" 2>/dev/null; then
    echo "  intact: $p"
  else
    echo "  MISSING (investigate before pushing!): $p"; fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo; echo "Verification FAILED -- do not push. Inspect ${MIRROR}"; exit 1
fi

cat <<EOF

--- Rewrite complete and verified, locally only. ---
Rewritten mirror: ${MIRROR}

Nothing has been pushed. When you have coordinated with collaborators and
fork owners, push deliberately from the mirror:

    git -C "${MIRROR}" push --force --all
    git -C "${MIRROR}" push --force --tags

Then, and only then:
  - Open a GitHub Support ticket to purge cached blob views + fork copies.
  - Ask the 5 fork owners to delete or re-fork.
  - Rebase open PRs #79 and #81.
  - Tell everyone with a clone to re-clone (old clones can re-push the blobs).
EOF
