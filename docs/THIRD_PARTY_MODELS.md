# Third-party models and priors

SMILify is MIT-licensed, but the **legacy quadruped path** depends on models and
priors from the Max Planck Institute (SMPL / SMAL / SMALST) that are released
under research licences which **do not permit redistribution**.

Those files were previously committed to this repository by accident
(see [issue #84](https://github.com/FabianPlum/SMILify/issues/84)). They have been
removed from version control and are now git-ignored. If you need them, download
them yourself from the upstream sources below and accept their licences directly.

**None of this is required for the actively-developed paths.** The SMIL/insect
models and the neural-inference pipeline (`smal_fitter/neuralSMIL/`) do not use
any of these files.

## What you may need, and where to get it

| File | Expected location | Source |
| --- | --- | --- |
| `basicModel_f_lbs_10_207_0_v1.0.0.pkl`<br>`basicmodel_m_lbs_10_207_0_v1.0.0.pkl` | `smal_model/smpl/models/` | [SMPL](https://smpl.is.tue.mpg.de/) — register, then download "SMPL for Python users" |
| `walking_toy_symmetric_pose_prior_with_cov_35parts.pkl` | `data/priors/` | [SMAL](https://smal.is.tue.mpg.de/) / [SMALify](https://github.com/benjiebob/SMALify) |
| `zebra_walking_symmetric_pose_prior_with_cov_35parts.pkl` | `data/priors/` | [SMALST](https://github.com/silviazuffi/smalst) (Zuffi et al.) |
| `my_smpl_data_00781_4_all.pkl`, `symIdx.pkl`, … | `smal/` | [SMAL](https://smal.is.tue.mpg.de/) — already git-ignored |

On Windows, the SMAL pickles need converting to a Windows-compatible encoding:

```powershell
./utilities/convert_smal_windows.ps1
```

This produces the `*_WIN.pkl` variants that `config.py` looks for.

## What is *ours* and stays in the repo

Several of our own parametric models use the SMPL **file format** and an `SMPL_`
naming prefix. They are **not** SMPL and are unaffected by the above:

- `3D_model_prep/SMPL_fit*.pkl`, `SMPL_STICK.pkl`, `smpl_ATTA.pkl` — our insect
  models (55/49 joints, insect anatomy: coxa, trochanter, mandibles, abdomen
  segments). SMPL itself has 6890 vertices and 24 joints; these do not.
- `3D_model_prep/SMIL_*.pkl`, `SMILy_*.pkl` — our ant, stick-insect and mouse models.
- `data/priors/unity_*` — generated from our own replicAnt synthetic data by
  [`data/priors/prepare_shape_prior.py`](../data/priors/prepare_shape_prior.py).

## The history purge (done 2026-07-16)

These files were not only untracked but purged from **all** git history. `master`
is now `626bb22`, and a fresh clone contains none of the licensed blobs.

The tooling is kept in [`scripts/purge_licensed_blobs.sh`](../scripts/purge_licensed_blobs.sh)
in case it is ever needed again:

```bash
./scripts/purge_licensed_blobs.sh            # dry run: report only
./scripts/purge_licensed_blobs.sh --execute  # rewrite a mirror clone locally
```

Neither mode writes to the remote — `--execute` rewrites a mirror clone and
prints the force-push commands to run deliberately.

A rewrite alone is **not** sufficient: GitHub still serves unreachable blobs by
SHA and from forks, so it must be followed by a GitHub Support ticket and
coordination with fork owners.

## What contributors and fork owners need to do

Because all history was rewritten, **every commit has a new SHA**. Old clones and
forks still contain the licensed blobs and will reintroduce them if pushed.

> **Back up your model files first.** These files are git-ignored now, but if
> your clone predates the purge they were *tracked*, so `git reset --hard` onto
> the rewritten history **deletes them from disk**. Copy them somewhere safe
> before you start, then copy them back afterwards.

### If you have a clone

Simplest and safest is to **re-clone**. To keep an existing clone instead:

```bash
cp -r smal_model/smpl/models /tmp/model-backup          # back up (see warning)
cp data/priors/*_pose_prior_with_cov_35parts*.pkl /tmp/model-backup/

git fetch origin
git checkout master
git reset --hard origin/master                          # deletes the files above
git reflog expire --expire=now --expire-unreachable=now --all
git repack -ad && git prune --expire=now                # `gc` alone is NOT enough

cp -r /tmp/model-backup/models smal_model/smpl/         # restore your copies
cp /tmp/model-backup/*.pkl data/priors/
```

`git gc --prune=now` does **not** remove the blobs — they sit in an existing
pack, and only `git repack -ad` rewrites it to evict them. Verify with:

```bash
git cat-file -e f5337df59c61c6825ec1dda302d38bfa09dcacc4 && echo "STILL PRESENT" || echo "purged"
```

Any local branch created before the purge still carries the blobs. Rebase it
onto the rewritten `master` (`git rebase --onto origin/master <old-base> <branch>`)
rather than merging, and never force-push a pre-purge branch.

### If you have a fork

A force-push to this repository does **not** touch forks, and GitHub keeps forks
in a shared object store — so blobs can stay reachable through a fork even after
the parent is clean. Please either:

- **Delete the fork** and re-fork if you still need it (cleanest), or
- Reset it onto the rewritten history and force-push your own fork:

  ```bash
  git remote add upstream https://github.com/FabianPlum/SMILify.git
  git fetch upstream
  git checkout master && git reset --hard upstream/master
  git push --force origin master
  ```

Open PRs raised from pre-purge branches must be rebased onto the new `master`
or closed and reopened; their old commits no longer exist upstream.

### Note on the contribution graph

The rewrite gives every commit a new SHA while preserving author dates, so
GitHub credits the rewritten commits *in addition to* the originals. Profile
commit counts are inflated as a result. This is cosmetic — no work was
duplicated or lost — and is inherent to any `filter-repo` history rewrite.

## Please don't re-add them

`.gitignore` now blocks these paths. Avoid blanket `git add -A`; prefer staging
explicit paths, and check `git status` before committing.
