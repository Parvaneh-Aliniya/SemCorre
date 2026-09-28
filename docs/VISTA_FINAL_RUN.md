# Vista final run — upload, batch, redraw (2026-03)

This pack includes **all code changes** for final review figures:

| Area | Files |
|------|--------|
| Pair / bidirectional figures | `scripts/interactive_correspondence.py` (neon GT/pred, IoU headers, hot-pink back line) |
| Exp5 chain (ROI → older only, no bridge) | `scripts/batch_mammo_correspondence.py` |
| Redraw pairs without GPU | `scripts/regenerate_pair_figures.py` |
| Exp3 small/large grids | `scripts/draw_exp3_group_compare.py` |
| Exp set definitions | `data/exp_sets/*.json` |
| SLURM jobs | `scripts/run_vista_exp*.slurm` |
| Post-batch regen helper | `scripts/vista_regen_figures.sh` |

**Not in git:** mammogram packs on `$SCRATCH/sk_review/...`, Python venv, HF cache, batch outputs.

---

## 1. PC — push code (preferred)

See **`docs/VISTA_GIT_SYNC.md`**.

```bash
cd "/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence"
git add scripts/ utils/ eval/ docs/ data/exp_sets/
git commit -m "Your message"
git push origin stats/benchmarks
```

---

## 2. Vista login — pull

```bash
cd /home1/11364/paliniya/projects/semcorre
bash scripts/vista_git_pull.sh
echo /scratch/11364/paliniya/stablekeypoints/sk_env > .semcorre_venv_path
bash scripts/vista_job_preflight.sh
```

**Legacy (no GitHub):** `bash scripts/vista_pack_for_transfer.sh` + scp tarball — avoid when git works.

---

## 3. (was unpack) — same as step 2

---

## 4. Submit final review batches (GPU)

Use **`gh`** queue, **`ASC26012`**. Each script uses `--gt-compare` (Gaussian + ROI-box subfolders) and `--skip-existing` where noted.

| Review set | SLURM script | Run tag(s) |
|------------|--------------|------------|
| Exp1 views | `scripts/run_vista_exp1_views.slurm` | `vista_exp1_views` |
| Exp1 ROI-box GT | `scripts/run_vista_exp1_views_roi.slurm` | `vista_exp1_views_roi_box` |
| Exp3 cross-patient | `scripts/run_vista_exp3_cross.slurm` | `vista_exp3_cross` |
| Exp3 ROI-box | `scripts/run_vista_exp3_cross_roi.slurm` | `vista_exp3_cross_roi_box` |
| Exp5 temporal transfer | `scripts/run_vista_exp5_temporal.slurm` | `vista_exp5_temporal_transfer`, `vista_exp5_temporal_own_roi` |
| Exp5 temporal ROI-box | `scripts/run_vista_exp5_temporal_roi.slurm` | `*_roi_box` variants |

Example:

```bash
mkdir -p logs
sbatch scripts/run_vista_exp1_views.slurm
sbatch scripts/run_vista_exp3_cross.slurm
sbatch scripts/run_vista_exp5_temporal.slurm
# optional ROI-box-only slurm twins
```

Outputs: `$SCRATCH/semcorre_batch_outputs/batch_experiments/<RUN_TAG>/`

---

## 5. After batch (login node, no GPU) — redraw figures

If jobs ran with **old** code, re-run step 3–4. If jobs already finished, redraw from saved `*_pair.json`:

```bash
cd /home1/11364/paliniya/projects/semcorre
source /scratch/11364/paliniya/stablekeypoints/sk_env/bin/activate
export SCRATCH=/scratch/11364/paliniya

for tag in vista_exp1_views vista_exp3_cross vista_exp5_temporal_transfer vista_exp5_temporal_own_roi; do
  bash scripts/vista_regen_figures.sh "$tag"
done
```

Exp5 **chain overviews** (ROI-start logic) need a **new** exp5 GPU run or rebuild from chain JSON if you add a rebuild command later; pair PNGs regen without LDM.

---

## 6. Download results to PC

```bash
# on PC — example for one tag
scp -r paliniya@vista.tacc.utexas.edu:/scratch/11364/paliniya/semcorre_batch_outputs/batch_experiments/vista_exp1_views \
  "/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence/outputs/batch_experiments/"
```

Exp3 summary PNGs: `<run-root>/exp3_group_compare/*.png`
