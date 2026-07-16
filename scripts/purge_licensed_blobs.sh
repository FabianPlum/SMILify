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
# Even --execute does NOT touch the remote. It rewrites a mirror clone and
# prints the force-push commands for you to run deliberately.
#
# STATUS: this purge was executed on 2026-07-16. master is now 626bb22 and a
# fresh clone carries none of the licensed blobs. The notes below are kept for
# anyone re-running it (e.g. if blobs are ever reintroduced).
#
# READ THIS FIRST -- a rewrite is NOT sufficient on its own:
#   1. Force-pushing rewrites every branch. Open PRs need rebasing and every
#      existing clone must re-clone. Coordinate first.
#   2. GitHub keeps unreachable blobs reachable by SHA and serves them from
#      forks. After force-pushing you MUST open a GitHub Support ticket asking
#      them to purge cached views, and ask fork owners to delete/re-fork.
#   3. Treat the SMPL/SMAL copies as having been public. If that matters
#      contractually, tell MPI rather than relying on a silent rewrite.
#   4. A force-push INFLATES the GitHub contribution graph: rewritten commits
#      keep their author dates but get new SHAs, so GitHub credits them again
#      on top of the originals. Cosmetic, but expect ~1 extra "commit" per
#      rewritten commit on your profile. There is no clean fix short of asking
#      GitHub Support.
#   5. Merge any pending PR BEFORE purging. Otherwise its base and head are both
#      rewritten and the PR breaks; and master ends up with the blobs stripped
#      but without the .gitignore guard that lives on the PR branch.
#
set -euo pipefail

# Source to clone. Override to rehearse the rewrite against a local copy
# without hitting the network, e.g.  PURGE_SOURCE=. ./scripts/purge_licensed_blobs.sh --execute
PURGE_SOURCE="${PURGE_SOURCE:-https://github.com/FabianPlum/SMILify}"

# Where a deliberate force-push would go. Kept separate from PURGE_SOURCE so a
# local-path rehearsal (PURGE_SOURCE=.) still prints the real remote.
PUSH_REMOTE="${PUSH_REMOTE:-https://github.com/FabianPlum/SMILify.git}"

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

# git-filter-repo deliberately removes the 'origin' remote after rewriting, to
# stop you force-pushing by reflex. Re-add it so the commands printed below
# actually run; this does NOT push anything.
git -C "$MIRROR" remote remove origin 2>/dev/null || true
git -C "$MIRROR" remote add origin "$PUSH_REMOTE"

cat <<EOF

--- Rewrite complete and verified, locally only. ---
Rewritten mirror: ${MIRROR}
Origin re-added : ${PUSH_REMOTE}  (nothing pushed)

Nothing has been pushed. When you have coordinated with collaborators and
fork owners, push deliberately from the mirror:

    git -C "${MIRROR}" push --force --all origin
    git -C "${MIRROR}" push --force --tags origin

Both are required: tags are branch archives here and pin old commits, so
pushing branches alone would leave the blobs reachable via tags.

The pushing clone needs credentials for the remote. A mirror cloned over
plain https in a bare WSL shell typically has none and will hang on a
credential prompt; push from wherever your git/gh auth already works.

Then, and only then:
  - Open a GitHub Support ticket to purge cached blob views + fork copies.
  - Ask the 5 fork owners to delete or re-fork.
  - Rebase open PRs #79 and #81.
  - Tell everyone with a clone to re-clone (old clones can re-push the blobs).
EOF
