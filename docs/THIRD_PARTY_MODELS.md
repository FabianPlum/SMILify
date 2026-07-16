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

## Purging them from git history

Untracking these files stops them being distributed *going forward*, but every
historical commit still carries the blobs, so they remain downloadable. Fully
removing them means rewriting all history:

```bash
./scripts/purge_licensed_blobs.sh            # dry run: report only
./scripts/purge_licensed_blobs.sh --execute  # rewrite a mirror clone locally
```

Neither mode writes to the remote — `--execute` rewrites a throwaway mirror
clone and prints the force-push commands for you to run deliberately.

A rewrite alone is **not** sufficient: GitHub still serves unreachable blobs by
SHA and from forks, so it must be followed by a GitHub Support ticket and
coordination with fork owners. The script prints the full checklist.

## Please don't re-add them

`.gitignore` now blocks these paths. Avoid blanket `git add -A`; prefer staging
explicit paths, and check `git status` before committing.
