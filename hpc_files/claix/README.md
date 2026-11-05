# CLAIX (RWTH Aachen) for Holotype training

This page covers running Holotype multi-view DDP training on RWTH's CLAIX cluster: access, storage, limits, the software environment, and the job scripts in this folder.

It was researched on 2026-11-03 from four sources:
- the RWTH IT Center help pages
- the RWTH HPC changelog and status portal
- RWTH's official examples repository
- Uness's CLAIX runs in September 2026: fork `Uness10/SMILify`, in `hpc_files/rwth/`, `cmnds*.txt` and `logs/`

Since 2026-11-04 these scripts have run from your account on project rwth2151: smoke job 5444995 on one node, and the 16-GPU run 5444996 on four. Items still marked **(unverified)** have not been checked yet.

| File | What it does |
|---|---|
| `train.sbatch` | Runs one wall-clock link of a multi-view DDP run on 1 or more c23g nodes. It resumes from the newest complete `checkpoint_epoch_*.pth` and exits at once if `final_model.pth` exists. |
| `submit_chain.sh` | Submits N links, each with `--dependency=afterany` on the previous one. |
| `rebase_config.py` | Rewrites a config's local absolute paths for CLAIX, sets keys with `--set`, and checks that all input files exist. |
| `environment_claix.yml` | The conda env recipe validated on CLAIX: channel-pinned torch/pytorch3d and pip pins matching the workstation (see [Software](#software)). |
| `ssh_config.example` | `~/.ssh/config` entries for the login, GPU-login and copy nodes, with connection sharing. |

## Cluster status when this was written (2026-11-03)

Check the [status portal](https://maintenance.itc.rwth-aachen.de/incident-change-application/status/messages/463) for the current state.


- **Long queues.** More nodes than usual have been down since 2026-11-03 15:15 (status portal), so expect long queue waits.
- **No SSH into job nodes.** Since 2026-07-09 you cannot SSH into a compute node running your job (incident 29866, no fix date). Neither VS Code nor a shell can attach to a running job. Watch jobs through their logs, `squeue`, `sacct` and the Grafana job view.
- **Maintenance windows:**
  - 2026-11-04 07:00–08:00: MySQL/JARDS; MFA is not affected.
  - 2026-11-10 07:00–07:30: MFA database; no new SSO logins during the window.

## First day, in order

1. **RegApp.** Log in once at https://regapp.itc.rwth-aachen.de through your home organisation (NHR Community AAI). This is mandatory, and unlinked accounts are being deactivated. Then:
   - Check that "Registered services → RWTH High Performance Computing" shows your HPC username.
   - Set your HPC password.
   - Create the RegApp TOTP token. It is separate from the RWTH IdM MFA token. Scan the QR code; the seed is not base32, so typing it in fails.
   - Upload an ed25519 public key and set its type to **interactive**.
2. **Network.** You can connect directly from:
   - the RWTH network, including RWTH eduroam
   - DFN-connected networks (most German research institutes). FZ Jülich works, also from home through the FZJ VPN's full tunnel (2026-11-04).
   - selected partner universities

   From anywhere else, connect to a VPN first: `vpn.rwth-aachen.de` for RWTH members, `vpn.hpc.rwth-aachen.de` for external HPC users. Both use Cisco Secure Client.
3. **SSH.** Merge `ssh_config.example` into `~/.ssh/config`, fill in your username, and run `ssh claix`. Enter your password and TOTP.
4. **Project limits.** Run these on the login node:
   ```
   r_wlm_usage -q -p <project-id>     # quota, allowed partitions, max walltime, max cores per job
   member show --name <project-id>    # confirms you are in the project group
   r_quota; r_quota -u <project-id>   # storage quotas
   ```
   Note your project type (RWTH-S, thesis, NHR or WestAI). Only NHR and WestAI projects may use c25g.
5. **Environment and backbone weights.** See [Software](#software).
6. **Upload.** Copy data and checkpoints over; see [Storage](#storage).
7. **Configs.** Rebase them with `rebase_config.py`; see [Jobs](#jobs).
8. **Smoke tests.** Run 1 node for 1 h, then 2 nodes with `NCCL_DEBUG=INFO`. Check the log lines listed under [Jobs](#jobs).
9. **Submit.** Time one epoch, work out how many links each run needs, then submit the chains.

## Connecting

CLAIX-2023 login nodes:

| Host | Use |
|---|---|
| `login23-1.hpc.itc.rwth-aachen.de` | General use. `-2` and `-4` are equivalent; `-3` still runs Rocky 8. |
| `login23-g-1.hpc.itc.rwth-aachen.de` | GPU tests on 4× H100. GPUs 0 and 1 are shared, GPU 2 is split into MIG slices, GPU 3 is exclusive to one process. Select GPUs by UUID (`nvidia-smi -L`). |
| `copy23-1.hpc.itc.rwth-aachen.de` | File transfer at 2×100 Gbit/s. No Slurm, no modules. |
| `login23-x-1`, `-x-2` | FastX desktops. Their CPU architecture differs from the compute nodes, so don't build environments there. |

Limits on every login node:
- 4 cores and 25 % of RAM per user. Above the RAM limit, your largest process is killed.
- CPU-heavy processes running over 20 minutes, and GPU processes over 30 minutes, may be killed without warning.
- Every login node reboots on Monday at 06:00. The copy nodes are excluded.

CLAIX-2025 has its own login nodes, `login25-1/-2/-3` and `login25-g-1`. Jobs submitted from there default to the c25 partitions, so always pass `-p`.

**Keys and MFA.**
- A key login is only accepted if you entered your password and TOTP within the last 10 hours. Whether that window is per node or cluster-wide is **unverified**.
- Keys expire after 12 months and cannot be re-added.
- Never set `PreferredAuthentications publickey`: it suppresses the password prompt, so you can't log in at all.

**VS Code.**
- RWTH has no guidance on VS Code. Remote-SSH to `claix` works like any SSH host, and with the shared connection from `ssh_config.example` it opens without a new prompt.
- The VS Code server and its extensions run on the login node, so the limits above apply to them.
- Keep data and run directories out of the file watcher: `"files.watcherExclude": {"**/runs/**": true, "**/*.h5": true}`.
- Training runs only through `sbatch`.

**Scheduling from Claude Code.**
- Keep Claude Code running locally in VS Code. After you open the master connection (`ssh claix`, then password and TOTP), Claude can run `ssh claix '...'` and `rsync ... claix-copy:...` itself for up to 10 hours.
- You type the password and TOTP. Never hand them to Claude: the terms of use make credentials personal.
- RWTH asks that `squeue`/`sinfo` be polled at most every 2 minutes.

**Monitoring commands:**
- `squeue --me`, or `r_squeue` for all of the project's jobs
- `sacct -X -j <id> -o AllocTRES%100`
- `seff <id>`
- Per-job GPU use in Grafana: https://perfmon.hpc.itc.rwth-aachen.de

**Other ways in:**
- JupyterHub: `https://jupyterhub.hpc.itc.rwth-aachen.de:9651`, up to 2 GPUs.
- FastX: its web servers have been stopped since 2026-10-22.

## Storage

| | Path | Quota (space / files) | Backup | Use |
|---|---|---|---|---|
| `$HOME` | `/home/<user>` (GPFS) | 150 GB / 1M | daily, plus snapshots | code, configs |
| `$WORK` | `/work/<user>` (GPFS) | 250 GB / 1M | snapshots only | |
| `$HPCWORK` | `/hpcwork/<user>` (Lustre, 500 GB/s) | 1 TB / ~1M | **none** | data, runs, environments |
| Node SSD | `$TMP`, or `$BEEOND` with `#SBATCH --beeond` | c23g 635 GB, c25g 1.6 TB | wiped when the job ends | staging |

**Project directories.**
- Paths: `/home/<project-id>`, `/work/<project-id>` and `/hpcwork/<project-id>`. Access is controlled by a group with the same name as the project (setgid, umask 007).
- A project gets the same default quota as a user.
- For more space, email servicedesk@itc.rwth-aachen.de with the project ID, filesystem, amount, number of files and duration.
- Nothing is purged. Project storage stays accessible for 8 months after the project ends.

**Suggested layout.**
- Put the environment and caches in your personal `$HPCWORK`. The env is about 120k files and 9 GB; the ViT-L weights are about 1.2 GB.
- Put data and runs in the project's `/hpcwork/<project-id>`.
- Keep every output directory off `$HOME`. Its quota cost Uness 13 epochs on 4 GPUs: job 3359198 died with `OSError [Errno 122]` inside `imageio.imsave`.

```
P=/hpcwork/<project-id>
$P/data/           # mirrors the workstation repo root: HDF5 files, 3D_model_prep/ (one --map covers both)
$P/code/<tag>/     # one pinned checkout per campaign; submit from here
$P/runs/           # mirrors /media/fabi/Data/SMILify_RUNS
```

**Upload** through the copy node with rsync over SSH. RWTH's rclone page warns against using more than 4 streams or 16 parallel transfers.
```
rsync -aP --mkpath combined_mice_falkner5_white_m103_v6.h5 claix-copy:/hpcwork/<project-id>/data/
rsync -aP --mkpath 3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_limbfix_proxies_limits.pkl \
      claix-copy:/hpcwork/<project-id>/data/3D_model_prep/
rsync -aP --mkpath /media/fabi/Data/SMILify_RUNS/long_schedule/full_combined_fk5_v6limits_twist/checkpoints/checkpoint_epoch_0099.pth \
      claix-copy:/hpcwork/<project-id>/runs/long_schedule/full_combined_fk5_v6limits_twist/checkpoints/
```

File sizes:
- combined HDF5: 19.9 GB
- M103 `proxies51_recal`: 6.2 GB
- Falkner skip-5 v6: 12.5 GB
- sticks `FIXED_raw2d`: 21.2 GB
- mouse `.pkl`: 10 MB
- one ViT-L multi-view checkpoint, including Adam state: 4.3 GB

The default Lustre layout already spreads files over 1 GiB across 4 storage targets, and files over 64 GiB across 16. The HDF5 files need no `lfs setstripe`.

**Checkpoints.** Saving every 2 epochs for 200 epochs comes to about 430 GB. Prune old periodic checkpoints by hand, and copy the ones worth keeping off `$HPCWORK`, which has no backup.

## Compute and limits

| | c23g (CLAIX-2023) | c25g (CLAIX-2025, pilot) |
|---|---|---|
| Who can submit | all projects | NHR and WestAI projects only |
| Nodes | 50 | 20 |
| GPUs | 4× H100 96 GB HBM2e, NVLink; power capped at 600 W since 2026-06-30 | 4× H100 SXM5 80 GB, NVLink |
| CPU / RAM | 2× Xeon 8468 (96 cores) / 512 GB | 2× EPYC 9535 (128 cores) / 786 GB |
| Per GPU | at most 24 cores and 122 GB | the docs give the same caps, apparently copied from c23g |
| Interconnect | NDR InfiniBand, about 25 GB/s per direction per node | Omni-Path CN5000; NCCL would need a plugin for it (inferred), which doesn't matter while jobs are capped at 1 node |
| Local SSD | 635 GB | 1.6 TB |
| Multi-node GPU jobs | yes | no, 1 node max during the pilot |
| Max walltime | set per project | 48 h (changelog 2026-10-19; the intro page still says 24 h) |

**Walltime and job limits.**
- `--time` defaults to 15 minutes.
- The maximum walltime is set per project (`r_wlm_usage -q -p`): 7 days for the free default project, and whatever was applied for on granted projects. Uness's `rwth2151` allows 24 h.
- Since 2026-03-06, c23g jobs longer than 24 h may use only half the nodes, so 24 h links are the practical choice either way.
- Granted projects can have 800 jobs running and 1000 submitted; the default project, 80 and 100.
- At the time limit a job is killed without warning. The trainer has no signal handler, so each link loses everything since its last checkpoint.

**Billing and budget.**
- Usage is billed in GPU-hours, for the largest share of a node you block (cores, memory or GPUs). One node for one hour costs 4 GPU-h; a 24 h link on 2 nodes costs 192 GPU-h.
- The most you can apply for:

  | Category | GPU-h |
  |---|---|
  | RWTH Small | 4k per year |
  | WestAI | 10k |
  | Lecture/Thesis | 1.5k |
  | NHR-Normal | 83k per year |

- Since 2026-08-18, new RWTH-S applications with an ML focus are redirected to WestAI.

**Quota rules.**
- The quota is monthly, but you can use up to 3× the monthly amount within any 3-month window.
- Beyond that, jobs run in `c23g_low`, which only starts jobs when the partition is otherwise idle.
- Using more than 6× the monthly amount within 4 weeks suspends submission.

**Scheduling.**
- Priority combines waiting time with a 2-week fair-share factor, and is only compared within a partition.
- Interactive shell: `salloc -p c23g -A <project-id> --gres=gpu:1 -n 24 -t 1:00:00`.
- The `devel` partition has no GPUs.

## Software

**Conda channels.** Use Miniforge. The firewall blocks repo.anaconda.com (Anaconda's `defaults` channel) but not anaconda.org, which hosts the `pytorch`, `nvidia`, `pytorch3d` and `conda-forge` channels.

**Build from `environment_claix.yml`, not the repo's `environment.yml`.** Under Miniforge the repo file builds a broken env (2026-11-04):
- **Wrong torch.** Miniforge's flexible channel priority installs conda-forge's torch 2.3.1. Its C++ ABI doesn't match the pytorch3d channel's 0.7.8 build, so `from pytorch3d import _C` fails with `undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs`. The recipe pins `pytorch::pytorch`, `pytorch::torchvision` and `pytorch3d::pytorch3d`.
- **Drifting pip packages.** The unpinned pip section installs numpy 2.2 and opencv 5.0. The recipe pins the workstation's versions (numpy 1.26.4, opencv-python 4.10.0.84, timm 1.0.20, ...), and `diagnostics/claix_env_PROBE` found the dataset decode byte-identical to the workstation with them.
- **MKL.** It pins `mkl<2024.1`; newer conda-forge MKL breaks pytorch-channel torch 2.3.1 (`iJIT_NotifyEvent`).

**H100 support.** The cu118 builds run natively on H100: `cuobjdump` shows sm_90 code in both cached packages, `pytorch-2.3.1-py3.10_cuda11.8_cudnn8.7.0_0` and `pytorch3d-0.7.8-py310_cu118_pyt231`. torch bundles NCCL 2.20.5.

**No modules needed.** You would only need modules for a source build, and then nvcc 11.8 requires GCC 11 or older; the cluster default is GCC 13.3.

```
# On login23-g-1, so that conda detects a GPU; or prefix the env create with CONDA_OVERRIDE_CUDA=11.8
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HPCWORK/miniforge3
source $HPCWORK/miniforge3/etc/profile.d/conda.sh
conda config --set auto_activate_base false
cd /hpcwork/<project-id>/code/<tag>
CONDA_OVERRIDE_CUDA=11.8 mamba env create -f hpc_files/claix/environment_claix.yml
conda activate pytorch3d
python -c "import torch, pytorch3d; from pytorch3d import _C; print(torch.__version__, torch.version.cuda)"
```
Deleting an old env on `$HPCWORK` takes several minutes (about 117k files).

**Backbone weights.** Compute nodes have no internet: Uness's runs show this, and RWTH's own examples download on login nodes. Fetch the weights on a login node:
```
export HF_HOME=$HPCWORK/hf_cache TORCH_HOME=$HPCWORK/torch_cache
python hpc_files/download_backbone_weights.py --backbones vit_large_patch16_224
```
`train.sbatch` uses the same `HF_HOME` and sets `HF_HUB_OFFLINE=1`.

**Shell.** The docs say the cluster supports only zsh, and RWTH's examples use `#!/usr/bin/zsh`. Bash job scripts, Uness's and these, run fine. A conda env that auto-activates from a shell rc file doesn't work in batch jobs; these scripts activate the env explicitly.

## Jobs

Code and configs:
- **Code and scripts.** The repo is public, so a login node can clone it without credentials:
  ```
  git clone https://github.com/FabianPlum/Holotype.git $P/code/<tag>
  git -C $P/code/<tag> log -1 --format=%h    # note the commit the campaign runs
  ```
- **Configs.** The run configs live under the untracked `diagnostics/`, so copy them over as well: `rsync -aP diagnostics/long_schedule_PROBE/configs/ claix-copy:$P/code/<tag>/configs_src/`.
- **Why a pinned checkout per campaign.** Each chain link `cd`s into the submit directory and runs whatever code is there when it starts, which can be hours after you submitted. So don't `git pull` in a checkout with a running chain. To move a campaign to newer code, clone a new checkout, copy its config over, and submit the next links from there; they resume from the same checkpoint folder, because the config's paths are absolute.
  - If the architecture changes, `load_checkpoint` drops the mismatched tensors and resets the epoch to 0. Its only messages are "Skipped N mismatched keys" and "Epoch counter reset to 0".

**Make cluster configs** on a login node:
```
cd $P/code/<tag>; mkdir -p logs configs_claix
python hpc_files/claix/rebase_config.py configs_src/full_combined_fk5_v6limits_twist.json configs_claix/combined_fk5.json \
    --map /home/fabi/dev/SMILify=$P/data \
    --map /media/fabi/Data/SMILify_RUNS=$P/runs \
    --set output.save_checkpoint_every=2
```
Set the checkpoint cadence in the config. The trainer ignores `--save_every_n_epochs` for configs with a `"mode"` key.

For the combined run, upload e99 into the run's own `checkpoint_dir`. `train.sbatch` then resumes from e99, because it is the newest checkpoint there; the config's own `resume_checkpoint` points at run 1's e49 and is no longer used.

**Blocked splits write to `*_fold<k>` folders.** With `split_method: blocked` the trainer appends `_fold<k>` to `checkpoint_dir`, `visualizations_dir` and `plots_dir` (k = `holdout_fold`, or the last fold when it is null: 4 with the 0.80/0.05/0.15 ratios), so the outputs of `.../checkpoints` land in `.../checkpoints_fold2`. `train.sbatch` resolves the folder the same way when it looks for the checkpoint to resume from.

**Smoke tests** need their own `checkpoint_dir`. Otherwise the smoke run's checkpoint becomes the real run's resume point.
```
python hpc_files/claix/rebase_config.py configs_claix/combined_fk5.json configs_claix/smoke.json \
    --set output.checkpoint_dir=$P/smoke/ckpt --set output.save_checkpoint_every=1 \
    --set training.resume_checkpoint=$P/runs/long_schedule/full_combined_fk5_v6limits_twist/checkpoints/checkpoint_epoch_0099.pth
CONFIG=configs_claix/smoke.json sbatch -A <project-id> --time=01:00:00 --export=ALL,CONFIG hpc_files/claix/train.sbatch
CONFIG=configs_claix/smoke.json NCCL_DEBUG=INFO sbatch -A <project-id> --nodes=2 --time=01:00:00 \
    --export=ALL,CONFIG,NCCL_DEBUG hpc_files/claix/train.sbatch
```

Check the logs:
```
grep -aE "Successfully initialized NCCL|gloo|NET/IB|NET/Socket|Skipped [0-9]+ mismatched|Epoch counter reset|DDP replicas after epoch" \
    logs/smil_train_<jobid>.out logs/smil_train_<jobid>.err
```
What a healthy run shows:
- One `Successfully initialized NCCL process group` line per rank, and no gloo fallback.
- On 2 nodes, `NET/IB` rather than `NET/Socket`.
- No `Skipped` or `Epoch counter reset` lines.
- `DDP replicas after epoch N: identical`.

To time an epoch, compare the checkpoint timestamps: `ls -l --time-style=+%H:%M $P/smoke/ckpt` (`$P/smoke/ckpt_fold<k>` for a blocked split).

**Measured speed**, both before #149 moved the image downscale into the loader workers:
- Smoke test (2026-11-05: one c25g node, 4x H100 80 GB, 4 samples per GPU, 6 views per sample): 0.675 s per training step, so 562 steps (one epoch of the combined dataset on 16 GPUs) take about 6.3 min plus about 20 s of validation. That is slower than the 0.43 s the AWS report estimated from the workstation; the step is limited by the host CPU, not the GPU.
- 16-GPU run 5444996 (four c23g nodes, 4 samples per GPU, 12 views per sample): 1.22 s per step, about 13.2 min per epoch including validation. Twice the images per GPU take 1.8× as long, so the step is still host-bound.

Compare runs on newer code against these numbers.

**Real runs.** Each run needs about `ceil(remaining_epochs × hours_per_epoch / 23)` links:
```
CONFIG=configs_claix/combined_fk5.json bash hpc_files/claix/submit_chain.sh 5 \
    -A <project-id> --export=ALL,CONFIG hpc_files/claix/train.sbatch
```

**Effective batch** is per-rank batch × ranks × gradient-accumulation steps. The combined run used 4 × 2 × 2 = 16. One node keeps 16 with `EXTRA_ARGS="--gradient_accumulation_steps 1"` (this CLI override is honoured for new-style configs). Two nodes at 16 would mean a batch of 2 per GPU. This is your decision; the scripts don't change it.

## From Uness's runs (September 2026, account rwth2151)

- **What worked.** c23g nodes `n23g00xx`; c25g nodes `n25g000x` and `s25g0007`. A conda env in `$HPCWORK` with torch 2.3.1 and CUDA 11.8. `srun` plus `torchrun` (c10d) across 2 nodes (8 ranks) initialised NCCL on every rank with our current `setup_ddp`, which is byte-identical to his. The rendezvous used the plain hostname and a fresh port, so the trainer's own TCPStore bound without conflict.
- **Speed.** Stick single-view ViT-L took about 13 min per epoch on one node, on both c23g and c25g. Stick multi-view, 25 epochs took about 2 h on one node (batch 8 per rank) and also on 2 nodes (batch 4 per rank). At a fixed effective batch, going to 8 GPUs gained nothing.
- **Caveat.** All his multi-GPU runs predate the DDP gradient-sync fix (#119/#120, merged 2026-11-02), so they trained unsynced replicas. Inter-node gradient all-reduce on CLAIX is therefore untested, and his timings include no all-reduce cost.
- **Walltime.** His 2-node runs were killed at the 24 h limit by SIGTERM, as expected. They saved checkpoints every 2 epochs.
- **A warning in his logs:** `Applied workaround for CuDNN issue, install nvrtc.so`. It affects performance only.

## Your project (rwth2151, checked 2026-11-04)

- **Limits** (`r_wlm_usage -q -p rwth2151`): a WestAI project; partitions c23g and c25g; 24 h max walltime; at most 384 cores per job, so at most 4 c23g nodes (16 GPUs) per job.
- **SSH.** The shared connection from `ssh_config.example` works: after one password and TOTP login, later `ssh` and `rsync` calls to the same host connect without a prompt.
- **Inter-node traffic.** The 16-GPU run's NCCL log shows `NET/IB` with GPUDirect RDMA.

Still open:
- Whether the 10-hour key window is per node or cluster-wide. RWTH doesn't document it.
- NVLink layout and number of InfiniBand cards per c23g node: run `nvidia-smi topo -m` and `ibstat` inside a job.
- The size of `/dev/shm` inside a job, which the DataLoader workers use.

## Trainer issues found along the way (not fixed)

- `train_multiview_regressor.py` ignores `--save_every_n_epochs` for configs with a `"mode"` key, because that flag is missing from the new-config override block (`if "mode" in _raw:` in `main()`).
- `setup_ddp` builds its own `TCPStore` on `MASTER_PORT`. That is fine under torchrun with torch 2.3.1, which hands workers a fresh port. Torch 2.4 and later probably share torchrun's own store on that port. After any upgrade, check the log for `Successfully initialized NCCL` rather than the gloo fallback.
- `setup_ddp` tries an InfiniBand hostname first by appending `i` to the host name, which is JSC's naming convention. On CLAIX it falls back to the plain host name.

## Sources

**Help pages** are at `https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/<id>/`; the last-changed date is in brackets.

| Topic | Article ID |
|---|---|
| filesystems | `da307ec2c60940b29bd42ac483fc3ea7` [2026-10-01] |
| project storage | `7d3f03d53d1a4387b90b522fda23833d` |
| quota | `adbc5a80a54e4bdb9db283933768c7a2` |
| node-local storage | `3fae2a161a514e9e90d020133d557500` |
| data transfer | `db3e5fd39d1d42c9815b4fa689719ac9` |
| login nodes, CLAIX-2023 | `0a23d513f31b4cf1849986aaed475789` [2026-10-01] |
| login nodes, CLAIX-2025 | `f92f2135151746908074f35cde2e0ef5` |
| SSH keys | `14e38a2d1975405d9fbf623a2237e552` [2026-09-22] |
| MFA login | `fb61d6c86ae245b5b7bba8c0cb7db6eb` |
| partitions | `9108f4a6f43c40a3a168919afd36839d` [2026-09-22] |
| hardware, CLAIX-2023 | `fbd107191cf14c4b8307f44f545cf68a` |
| hardware, CLAIX-2025 | `0ea15c530ac94776a8e65325b55ce110` |
| CLAIX-2025 intro | `783f1418454e4cac84e4a230060b351b` [2026-10-28] |
| resources | `d99e9f242bdb48eaa1c63c45e84fc32f` |
| job parameters | `3d20a87835db4569ad9094d91874e2b4` |
| accounting | `f37ad962ddce4790925ffc45d8a5a90f` |
| core and GPU hours | `fed436424d464e01ac7add680162a852` |
| project categories | `45825b06afb647e194be4a5b9f5b8768` |
| interactive and testing | `5b115949502a40c0819005719a70e49e` |
| Python and conda | `7230ed5050e94aacbbe1db14cada5b56` [2026-07-22] |
| Apptainer | `e6f146d0d9c04d35aeb98da8d261e38b` |
| job monitoring | `3a11a76fdf30476bb4b1a8b30661dab3` |

**Changelog posts** are at `https://blog.rwth-aachen.de/itc-changes/en/<path>/`:
- `2026/03/09/runtime-limits-for-gpu-jobs`
- `2026/06/24/ml-compute-quota-now-shown-in-gpu-h`
- `2026/06/30/gpu-tdp-reduced-on-claix-2023-nodes`
- `2026/07/06/english-important-changes-for-hpc-accounts-usage-of-the-nhr-community-aai`
- `2026/07/15/changes-to-default-partition-for-claix-jobs-that-do-not-define-it`
- `2026/08/05/important-changes-to-rwth-s-project-application-procedures-with-ai-ml-focus`
- `2026/10/19/time-limit-of-claix-c25-partition-extended`

**Other sources:**
- Status portal: https://maintenance.itc.rwth-aachen.de/incident-change-application/status/messages/463 (incident 29866)
- RWTH examples repository: https://git-ce.rwth-aachen.de/high-performance-computing/examples (`job-script-examples/`, `pytorch/cifar10_distributed/`)
- ResNet-50 example: https://github.com/RWTH-HPC/ResNet50-PyTorch
- aiXcelerate 2025 slides on storage and I/O: https://blog.rwth-aachen.de/itc-events/files/2025/12/
