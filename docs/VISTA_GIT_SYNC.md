# Vista: sync code with Git (preferred)

Use **git push** from your PC and **git pull** on Vista instead of tarballs / scp for `scripts/`, `utils/`, `eval/`, and `data/exp_sets/`.

**Repo:** https://github.com/Parvaneh-Aliniya/SemCorre.git  
**Branch used for Vista work:** `stats/benchmarks` (confirm with `git branch` on both sides).

**Vista project path:** `/home1/11364/paliniya/projects/semcorre`

Large data stays on **`$SCRATCH`** (packs, batch outputs, HF cache) — never in git.

---

## One-time: Vista clone (if not already a git repo)

```bash
cd /home1/11364/paliniya/projects
# if semcorre exists only as unpacked files, rename backup first:
# mv semcorre semcorre_backup_YYYYMMDD
git clone https://github.com/Parvaneh-Aliniya/SemCorre.git semcorre
cd semcorre
git checkout stats/benchmarks
```

If `semcorre` is already a clone, skip clone and only pull below.

GitHub auth on Vista: use a **personal access token** as password, or SSH remote `git@github.com:Parvaneh-Aliniya/SemCorre.git` with key loaded.

---

## Every code update

### On PC (Git Bash)

```bash
cd "/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence"
git status
git add scripts/ utils/ eval/ docs/ data/exp_sets/
git commit -m "Describe your figure/batch fix"
git push origin stats/benchmarks
```

Commit only source and configs — not `deploy/*.tar.gz`, not `data/roi_overlays_*` PNG trees.

### On Vista (login)

```bash
cd /home1/11364/paliniya/projects/semcorre
bash scripts/vista_git_pull.sh
```

Or manually:

```bash
git fetch origin
git pull origin stats/benchmarks
for f in scripts/*.slurm scripts/*.sh; do [[ -f "$f" ]] && sed -i 's/\r$//' "$f"; done
```

---

## Regenerate figures after pull (login)

Login needs modules for the scratch venv:

```bash
module load gcc python3 2>/dev/null || true
source /scratch/11364/paliniya/stablekeypoints/sk_env/bin/activate
python -c "import torch; print('OK')"

export RUN=$SCRATCH/semcorre_batch_outputs/batch_experiments/vista_exp3_cross_roi_box
export PACK=$SCRATCH/sk_review/roi_overlays_exp3_cross
cd ~/projects/semcorre
python scripts/regenerate_pair_figures.py --run-root "$RUN" --pack-dir "$PACK"
```

If `import torch` fails on login, use a short **gh-dev** job (see `docs/VISTA_FINAL_RUN.md`).

---

## Legacy: tarball

`bash scripts/vista_pack_for_transfer.sh` remains for offline transfer only. Prefer **git pull** when GitHub is reachable from Vista.
