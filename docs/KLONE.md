# Working on Klone (UW Hyak) — operating notes

Notes for anyone (human or agent) picking this project up on **klone**, UW's
condo-style Slurm cluster (Rocky 8, GPFS). Written 2026-11-15.

Each fact is tagged with where it came from:

- **[live]** — observed on the cluster on 2026-11-15; the command is given so it
  can be re-checked. Live Slurm output always beats this file.
- **[docs]** — from <https://hyak.uw.edu/docs/> (read 2026-11-15).
- **[policy]** — from the site policy in `/etc/claude-code/CLAUDE.md`.

Numbers go stale. Re-run the command before relying on one.

---

## 1. The rules that matter most

1. **Login nodes (`klone-login01/02`) are for editing, git, job submission and
   downloads only.** No training, no evaluation, no environment builds, nothing
   multi-threaded. `arbiter2` watches and enforces this. [docs][policy]
2. **Agent subagents must not be spawned from a login-node session** — the
   session gets killed by the per-user cgroup limit (see `AGENTS.md`). See §8.
3. **Never size anything from `nproc` / `free` / `os.cpu_count()`.** Those
   describe the whole shared node. Use `$SLURM_CPUS_PER_TASK` and
   `$SLURM_MEM_PER_NODE`; default to 1 thread if unset. [policy]
4. **Nothing on klone is backed up**, and `/gscratch/scrubbed` deletes files
   after 21 days without access. **This repository lives in scrubbed**, so
   GitHub is its only durable copy: commit and push often. [docs]
5. Request what the job needs, not the maximum. Over-requesting lowers the
   group's scheduling priority. [policy]

## 2. What this user can submit to

`hyakalloc` and
`sacctmgr -P show assoc where user=$USER format=Account,Partition,QOS` [live]:

| Account | Partition | Group total (CPU / mem / GPU) | Notes |
|---|---|---|---|
| `amath` | `gpu-rtx6k` | 40 / 376G / **8** | Was entirely free on 2026-11-15. **Default choice for this project.** |
| `amath` | `gpu-l40s` | 32 / 377G / 2 | 1 GPU free on 2026-11-15. |
| `amath` | `cpu-g2` | 320 / 2519G / 0 | CPU-only work. |
| `stf` | `gpu-l40`, `gpu-l40s` | 8 and 10 GPUs | Fully used on 2026-11-15; shared by a large club. |
| `stf` | `compute`, `cpu-g2`, `compute-hugemem`, `cpu-g2-mem2x` | — | CPU-only. |
| any | `ckpt`, `ckpt-g2`, `ckpt-all` | idle resources cluster-wide | Preemptible, see §4. |

- Limits are **per group, cumulative**, not per user: a GPU I hold is a GPU a
  labmate cannot get. [docs]
- Partition `MaxTime` is `UNLIMITED` and `DefaultTime` is 1 h
  (`scontrol show partition gpu-rtx6k`) [live], so **always pass `--time`**.
  Jobs that would run past the next maintenance window stay pending with
  `ReqNodeNotAvail`; `hyakalloc` prints the maintenance notice when one is
  scheduled. [docs]
- `DefMemPerCPU` is 1024 MB (`scontrol show config`) [live], so **always pass
  `--mem`**; the default is far too small for PyTorch.
- QoS `normal`: max 1000 submitted jobs per user. `ckpt*`: 250 running / 500
  submitted per user. `MaxArraySize` is 1001.
  (`sacctmgr -P show qos format=Name,Priority,MaxJobsPU,MaxSubmitPU`) [live]

### GPU hardware

| Partition | GPU | VRAM | Node shape |
|---|---|---|---|
| `gpu-rtx6k` | Quadro RTX 6000 | **24 GB** [live: `nvidia-smi` on g3026] | 40 CPU, 385 GB, 8 GPU |
| `gpu-l40s` | L40S | 48 GB [docs] | 128 CPU, 1.5 TB, 8 GPU |
| `gpu-l40` | L40 | 48 GB [docs] | — |
| `gpu-a40` | A40 | 48 GB [docs] | ckpt only for us |
| `gpu-a100` | A100 | 40 GB [docs] | ckpt only for us |
| `gpu-2080ti` | 2080 Ti | 11 GB [docs] | ckpt only for us |

The docs' GPU page lists the RTX 6000 as 48 GB; the nodes in `gpu-rtx6k`
report 24576 MiB. Trust `nvidia-smi`.

Fair share of a node per GPU on `gpu-rtx6k` is 5 CPUs and ~47 GB. Small
CIFAR-10 models need far less: `-c 4 --mem=16G` is the working default.

## 3. Requesting compute

Verified working from a non-interactive shell on the login node [live]:

```bash
# One-off command on a GPU node; blocks until it finishes, output comes back inline.
srun -A amath -p gpu-rtx6k --gpus=1 -c 4 --mem=16G -t 10 --job-name=smoke \
     bash -lc 'nvidia-smi; echo $SLURM_JOB_ID'
```

The allocation took a few seconds on 2026-11-15.

- **`srun … bash -lc '…'`** — short checks and builds where the output is wanted
  immediately.
- **`sbatch script.slurm`** — everything long-running. Preferred. Returns at
  once; poll with `squeue -u $USER` at intervals of tens of seconds, never in a
  tight loop.
- **`salloc`** — opens an interactive shell; of no use to a non-interactive
  agent.
- Job arrays (`#SBATCH --array=0-9%4`) for seed sweeps; `%4` caps concurrency so
  the group's 8 GPUs are not all taken at once.

Minimal batch template:

```bash
#!/bin/bash
#SBATCH --job-name=ar-train
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out   # the directory must already exist
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
# activate the environment, then run
```

Monitoring: `squeue -u $USER`; `sacct -j <id> -o JobID,State,Elapsed,MaxRSS,ReqMem`;
`scontrol show job <id>`; `scancel <id>`.

Common failures [docs]:

- `Exceeded job memory limit` / exit 137 — raise `--mem`. Do not blindly retry.
- `Invalid account or account/partition combination` — `-A` takes the *group*
  (`amath`), not the NetID.
- A missing `--output` directory makes the job fail instantly with no log.

## 4. Checkpoint partitions (`ckpt`, `ckpt-g2`, `ckpt-all`)

Free idle capacity across the whole cluster, including GPUs the group does not
own (104 idle GPUs on 2026-11-15, `hyakalloc`). [live]

- Jobs are **preempted without notice** when an owner wants the hardware, and
  are also requeued on a timer: every 4–5 h for CPU jobs, 8–9 h for GPU jobs.
  The job restarts from the top of its script until `--time` is used up. [docs]
- Any job sent there must save state periodically and resume from it.
- GPU syntax differs: `--partition=ckpt-all --gpus-per-node=a40:1`. [docs]
- `ckpt-all` mixes g1 (Intel) and g2 (AMD Genoa) nodes and several GPU models,
  so runs are not bit-reproducible across requeues. Pin a GPU type when that
  matters.
- Under heavy filesystem load ckpt is throttled; the last line of `hyakalloc`
  says so when it is. [docs]

Use the owned `amath` GPUs for anything interactive or timing-sensitive, and
ckpt for wide, resumable sweeps.

## 5. Storage

`hyakstorage` on 2026-11-15 [live]; the report refreshes about hourly, so do not
poll it.

| Path | Quota | State on 2026-11-15 | Use for |
|---|---|---|---|
| `/mmfs1/home/sgvtc` | 10 GB, 256k files | 6 GB (60%) | Config only. Never caches, envs, data. |
| `/gscratch/amath` | 15.4 TB, 15M files | 82% disk, **94% of inodes** | Things that must survive: the environment, final models, key results. Keep the file count low. |
| `/gscratch/stf` | 61 TB | **99% full** | Do not write here. |
| `/gscratch/scrubbed/sgvtc` | 10 TB soft, shared 100 TB | — | This repo, datasets, caches, bulky intermediate checkpoints. |
| `/tmp` = `/scr` on compute nodes | 358 GB node-local SSD | 355 GB free on g3026 | Heavy temporary I/O inside a job. Clean up afterwards. `$TMPDIR` is `/tmp`. |
| `/mmfs1/data` | read-only | — | Hosted datasets (ImageNet, Kinetics, …). **CIFAR-10 is not there** (`ls /mmfs1/data`). |

- `/gscratch` and `/mmfs1/gscratch` are the same place. `~/gscratch` is a
  symlink to `/gscratch/scrubbed/sgvtc`.
- **Scrubbed deletes any file not accessed for 21 days, file by file.**
  Touching files to dodge this is a policy violation. Consequences:
  - A virtualenv in scrubbed rots piecemeal and breaks in confusing ways. Keep
    the environment in `/gscratch/amath/sgvtc/`.
  - Anything an assessor needs later goes to GitHub (code, small results) or
    `/gscratch/amath/sgvtc/` (models).
- Scrubbed is world-readable by default.
- GPFS dislikes many small files and metadata scans. Store datasets as a few
  large array files, never as tens of thousands of images, and do not run
  `find` / `du` / `grep -r` over large trees. For unavoidable many-file
  datasets, pack a SquashFS image and mount it with `squashfuse` inside the
  job. [docs]

## 6. Software

- **Modules exist only on compute nodes** (`module avail` inside a job). [live]
  Available there: `cuda/11.2 … 13.4`, `gcc/9 … 15`, `cmake`,
  `conda/Miniforge3-25.9.1-0`, `apptainer`, `squashfuse`, `ngc`.
- System Python is 3.6.8 — do not use it for project code. [live]
- **`uv` 0.11.25 is installed at `~/.local/bin/uv`** and is visible on compute
  nodes. It is this project's environment manager.
- The shell's conda setup is broken: `.bashrc` points at a `~/miniconda3` that
  does not exist, and `~/.condarc` sets `envs_dirs`/`pkgs_dirs` to `.`. If conda
  is ever needed, use `module load conda` with an explicit `--prefix`. [live]
- Login shells print `…/toolchain-tensat/cargo/env: No such file or directory`.
  It is a stale line in `.bashrc`/`.bash_profile` and is harmless.
- Apptainer is available for containers. Build with
  `APPTAINER_CACHEDIR=/tmp` on a compute node, never in home. [docs]
- No `sudo`, and no `gh` CLI; use plain `git` over HTTPS (credentials are
  already stored).

### Environment rules for this project

- Build and install **inside a Slurm job**, not on the login node. [policy]
- `export UV_CACHE_DIR=/gscratch/scrubbed/sgvtc/uv-cache`. The default cache is
  in home and a CUDA PyTorch install would exceed the 10 GB quota.
- `export UV_LINK_MODE=copy` — the cache and the environment are on different
  filesets, so hardlinks are not possible.
- PyTorch wheels bundle their own CUDA runtime; `module load cuda` is not
  needed for them.

## 7. Network

- Compute nodes reach the internet through a shared proxy (`http_proxy` and
  `https_proxy` are preset). pypi.org and github.com returned 200 from g3026.
  [live]
- Compute nodes are on 1 GbE and the proxy is shared by the whole cluster: keep
  connections few and sequential. Do large downloads from the login node
  (10 GbE). [policy]
- Check the destination quota before any large download.

## 8. Running agents and subagents

- This Claude Code session normally runs on a login node. Agent/subagent/
  workflow fan-out from there gets the session killed.
- The `claude` CLI is at `~/.local/bin/claude` and is on `PATH` on compute
  nodes, and `api.anthropic.com` is reachable through the proxy. [live]
- So heavy agent work belongs inside a Slurm allocation — either start the whole
  session there, or run a headless `claude -p` under `srun`.

## 9. Pre-flight checklist for a job

1. `hostname -s` — on a login node? Then submit, do not run.
2. `hyakalloc` — is the group's GPU free, and is a maintenance window coming?
3. `--account`, `--partition`, `--gpus`, `-c`, `--mem`, `--time` all explicit.
4. Thread counts and DataLoader workers derived from `$SLURM_CPUS_PER_TASK`.
5. Log directory exists; outputs go to scrubbed (bulky) or amath (to keep).
6. The job records hostname, GPU model, job ID and start time in its log, for
   `JOURNAL.md`.
7. Sent to ckpt? Then it must checkpoint and resume.
