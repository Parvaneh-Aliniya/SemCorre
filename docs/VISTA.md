# Vista (TACC): full exp1–5 batch

**Login:** `ssh paliniya@vista.tacc.utexas.edu`  
**Project root:** `/home1/11364/paliniya/projects/semcorre`  
**Allocation:** `ASC26012`  
**Python env:** `/scratch/11364/paliniya/stablekeypoints/sk_env` (a **venv**, not conda)

Jobs `1024624` and `1027688` died in ~6s because preflight found no env. `conda activate sk_env` fails on Vista (`conda: command not found`). `echo "$CONDA_PREFIX" > .semcorre_venv_path` then writes an empty path; `test -x "$(cat .semcorre_venv_path)/bin/python"` is a false OK because that becomes `/bin/python`.

---

## Critical: GPU queue

| Queue | Hardware | SemCorre |
|-------|----------|----------|
| **`gh`** | Grace **Hopper** (H200 GPU) | **Use this** |
| `gh-dev` | Hopper (short tests) | Smoke tests only (2 h limit) |
| **`gg`** | Grace **CPU only** (no CUDA) | **Wrong** — job will hang or run on CPU forever |

Do **not** use `#SBATCH -p gg` for Stable Diffusion / CUDA.

Vista docs: [Vista user guide](https://docs.tacc.utexas.edu/hpc/vista/) — no `--gres`; one full node per job.

---

## Deep checklist (before `sbatch`)

On Vista login:

```bash
cd /home1/11364/paliniya/projects/semcorre
bash scripts/vista_job_preflight.sh
```

This checks: SLURM placeholders, **gg vs gh**, CRLF, code tree, **roi_overlays_cancer5**, venv + imports, disk, HF cache.

Point the job at the existing SK venv **before** `sbatch` (or the next job dies like 1027688):

```bash
echo /scratch/11364/paliniya/stablekeypoints/sk_env > .semcorre_venv_path
test -x /scratch/11364/paliniya/stablekeypoints/sk_env/bin/python && echo OK_PYTHON
```

Fix **one** `#SBATCH -A` and **one** `#SBATCH -p` only:

```bash
grep -E '^#SBATCH -(A|p)' scripts/run_vista_exp_all_1to5.slurm
# must show only:
#   #SBATCH -A ASC26012
#   #SBATCH -p gh
```

---

## Storage (home vs scratch)

Your **home** quota is small (~23 GB). The job script uses:

- `HF_HOME=$SCRATCH/hf_cache_semcorre` (Stable Diffusion download)
- `SEMCRE_OUT=$SCRATCH/semcorre_batch_outputs` (all batch PNGs/JSON)

Results path on Vista: **`$SCRATCH/semcorre_batch_outputs/batch_experiments/vista_exp_all_1to5/`**

Fetch with:

```bash
scp -r paliniya@vista.tacc.utexas.edu:/scratch/11364/paliniya/semcorre_batch_outputs/batch_experiments/vista_exp_all_1to5 \
  "/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence/outputs/batch_experiments/"
```

(Adjust if `echo $SCRATCH` on Vista differs.)

---

## Windows → Vista deploy (Git — preferred)

**PC:** commit + `git push origin stats/benchmarks`  
**Vista:** `cd ~/projects/semcorre && bash scripts/vista_git_pull.sh`  

Full steps: **`docs/VISTA_GIT_SYNC.md`**.

**Legacy tarball:** `bash scripts/vista_pack_for_transfer.sh` + scp (if GitHub unreachable from Vista).

**Vista login after pull:**

```bash
cd /home1/11364/paliniya/projects/semcorre
for f in scripts/*.slurm scripts/*.sh; do [[ -f "$f" ]] && sed -i 's/\r$//' "$f"; done

# Python: existing venv on scratch (NOT conda — `conda` is not on Vista PATH).
# Wrong: conda activate sk_env   → writes empty CONDA_PREFIX and jobs die in 6s.
module load gcc cuda python3 2>/dev/null || true
echo /scratch/11364/paliniya/stablekeypoints/sk_env > .semcorre_venv_path
test -x /scratch/11364/paliniya/stablekeypoints/sk_env/bin/python && echo OK_PYTHON
# Only if that python is missing: bash scripts/vista_setup.sh && echo "$PWD/venv" > .semcorre_venv_path

huggingface-cli login   # first job downloads SD 1.4 unless already in $SCRATCH/hf_cache_semcorre

bash scripts/vista_job_preflight.sh   # must say PREFLIGHT PASSED before sbatch
sbatch scripts/run_vista_exp_all_1to5.slurm
squeue -u $USER
tail -f logs/semcorre_1to5_<JOBID>.out
```

**48 h wall limit:** if the batch does not finish, run `sbatch` again — same `RUN_TAG` + `--skip-existing` continues where it left off.

### Smallest ROIs on the full EMBED table (not just cancer5)

The short job picked from the **5-patient pack only**. To rank every image with `ROI_coords`:

```bash
export EMBED_DATA_DIR=/scratch/11364/paliniya/embed_dataset/tables
python scripts/select_smallest_embed_rois.py --top 25
```

### Short ~1 h run (exp5 → 3 → 2 → 1, skip 4)

Cancel the full 75-pair job first if it is still running: `scancel 1029339`

Copy updated `scripts/batch_mammo_correspondence.py` and `scripts/run_vista_exp_short_1h.slurm` to Vista, then:

```bash
cd /home1/11364/paliniya/projects/semcorre
sed -i 's/\r$//' scripts/run_vista_exp_short_1h.slurm scripts/batch_mammo_correspondence.py
sbatch scripts/run_vista_exp_short_1h.slurm
```

Results: `$SCRATCH/semcorre_batch_outputs/batch_experiments/vista_short_1h/`
(exp5 folder appears first so you can `scp` it while 3/2/1 are still running).

---

## Optional: GPU smoke test (before full batch)

```bash
idev -p gh-dev -N 1 -n 1 -t 0:30:00
cd /home1/11364/paliniya/projects/semcorre
source /scratch/11364/paliniya/stablekeypoints/sk_env/bin/activate
module load gcc cuda python3
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Scripts reference

| File | Role |
|------|------|
| `vista_job_preflight.sh` | Validate environment before submit |
| `run_vista_exp_all_1to5.slurm` | Full exp1–5, `-p gh`, scratch outputs |
| `vista_setup.sh` | venv + PyTorch (cu129) + deps |
| `run_vista_subset.slurm` | Small subset test only |
