"""
Regenerate TPS figures (ROI + breast controls) from an existing exp5 batch run.
No Stable Diffusion — reuses SemCorre keypoints from chain.json or step TPS meta.

Example (3 pairs, ONE chain, NEW output folder — ROI + uniform breast TPS):
  python scripts/tps_only_exp5_from_batch.py \\
    --batch-run-dir outputs/batch_experiments/exp5_31292781_45209155_rmlo \\
    --exp5-views 31292781:R:MLO \\
    --exp5-max-steps 3 \\
    --tps-modes roi,breast \\
    --out-dir outputs/tps_new/31292781_rmlo_3pairs

Without --out-dir, PNGs overwrite the old batch step folders (easy to mistake for "nothing new").
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import batch_mammo_correspondence as batch  # noqa: E402
from utils import tps_roi_warp  # noqa: E402
from utils.tps_roi_warp import ControlMode  # noqa: E402

CHAIN_DIR_RE = re.compile(
    r"^p(?P<pid>\d+)_(?P<lat>[lr])_(?P<view>[a-z]+)_chain_from_(?P<anchor>.+)$",
    re.IGNORECASE,
)
STEP_DIR_RE = re.compile(
    r"^step_(?P<step>\d+)_(?P<anchor>\d{4}-\d{2}-\d{2})_to_(?P<trg>\d{4}-\d{2}-\d{2})$"
)


def parse_tps_modes_local(spec: str) -> tuple[ControlMode, ...]:
    raw = [x.strip().lower() for x in spec.split(",") if x.strip()]
    modes: list[ControlMode] = []
    for x in raw:
        if x not in ("roi", "breast"):
            raise SystemExit(f"Unknown --tps-modes entry {x!r} (use roi, breast)")
        if x not in modes:
            modes.append(x)  # type: ignore[arg-type]
    return tuple(modes)  # type: ignore[return-value]


def anchor_date_from_slug(slug: str) -> str:
    parts = slug.split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        return "-".join(parts[:3])
    return slug.replace("_", "-")


def load_kp_from_step_meta(step_dir: Path) -> tuple[tuple[float, float], tuple[float, float]] | None:
    metas = sorted(step_dir.glob("*_tps_*_warp_meta.json"))
    for meta_path in metas:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        src = data.get("semcorre_source_xy512") or data.get("method_src_xy512")
        dst = data.get("semcorre_target_xy512") or data.get("method_dst_xy512")
        if src and dst and "x" in src and "y" in dst:
            return (float(src["x"]), float(src["y"])), (float(dst["x"]), float(dst["y"]))
    return None


def file_stem_for_step(step_dir: Path) -> str:
    metas = sorted(step_dir.glob("*_tps_*_warp_meta.json"))
    if metas:
        return json.loads(metas[0].read_text(encoding="utf-8")).get("file_stem") or metas[0].stem.rsplit(
            "_tps_", 1
        )[0]
    corrs = sorted(step_dir.glob("*_correspondences_estimated.png"))
    if corrs:
        return corrs[0].name.replace("_correspondences_estimated.png", "")
    m = STEP_DIR_RE.match(step_dir.name)
    if m:
        return batch.slugify(f"step{int(m.group('step')):02d}_{m.group('trg')}")
    return step_dir.name


def discover_chain_dirs(exp5_root: Path) -> list[Path]:
    dirs = [p.parent for p in exp5_root.glob("*/*_chain.json")]
    dirs.extend(p for p in exp5_root.glob("p*_chain_from_*") if p.is_dir())
    seen: set[Path] = set()
    out: list[Path] = []
    for d in sorted(dirs):
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def steps_from_chain_json(chain_dir: Path, max_steps: int) -> list[dict]:
    chain_jsons = list(chain_dir.glob("*_chain.json"))
    if not chain_jsons:
        return []
    data = json.loads(chain_jsons[0].read_text(encoding="utf-8"))
    steps = list(data.get("steps") or [])
    if max_steps > 0:
        steps = steps[:max_steps]
    return steps


def steps_from_step_dirs(
    chain_dir: Path,
    pack_root: Path,
    pid: str,
    lat: str,
    view: str,
    anchor_date: str,
    max_steps: int,
) -> list[dict]:
    step_dirs = sorted(
        (p for p in chain_dir.iterdir() if p.is_dir() and STEP_DIR_RE.match(p.name)),
        key=lambda p: int(STEP_DIR_RE.match(p.name).group("step")),  # type: ignore[union-attr]
    )
    if max_steps > 0:
        step_dirs = step_dirs[:max_steps]
    out: list[dict] = []
    prev_trg: Path | None = None
    for step_dir in step_dirs:
        m = STEP_DIR_RE.match(step_dir.name)
        if not m:
            continue
        step_i = int(m.group("step"))
        trg_date = m.group("trg")
        if step_i == 1:
            src_path = batch.resolve_exam_image(pack_root, pid, anchor_date, lat, view)
        else:
            src_path = prev_trg
        trg_path = batch.resolve_exam_image(pack_root, pid, trg_date, lat, view)
        if src_path is None or trg_path is None:
            print(f"  skip {step_dir.name}: missing image path")
            continue
        kp = load_kp_from_step_meta(step_dir)
        if kp is None:
            print(f"  skip {step_dir.name}: no SemCorre keypoints (finish SemCorre for this hop first)")
            continue
        (method_src, method_dst) = kp
        out.append(
            {
                "step": step_i,
                "source_path": str(src_path),
                "target_path": str(trg_path),
                "source_kp_512": {"x": method_src[0], "y": method_src[1]},
                "target_kp_512": {"x": method_dst[0], "y": method_dst[1]},
                "target_exam": trg_date,
                "_step_dir": step_dir,
            }
        )
        prev_trg = trg_path
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TPS-only (roi/breast) for existing exp5 batch output.")
    p.add_argument(
        "--batch-run-dir",
        type=str,
        required=True,
        help="Folder from batch_mammo_correspondence, e.g. outputs/batch_experiments/exp5_62877247_lcc",
    )
    p.add_argument(
        "--pack-dir",
        type=str,
        default="../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5",
    )
    p.add_argument("--exp5-max-steps", type=int, default=3, help="Max hops per chain (default 3 pairs)")
    p.add_argument(
        "--exp5-views",
        type=str,
        default="",
        help="Only these chains, e.g. 62877247:L:CC or 31292781:R:MLO (empty = all chains in batch dir)",
    )
    p.add_argument("--tps-modes", type=str, default="roi,breast")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Write NEW TPS PNGs here (recommended). If empty, overwrites inside --batch-run-dir.",
    )
    args = p.parse_args()

    run_root = Path(args.batch_run_dir).expanduser().resolve()
    pack_root = Path(args.pack_dir).expanduser().resolve()
    modes = parse_tps_modes_local(args.tps_modes)

    exp5_root = run_root / "exp5_sequential_prior"
    if not exp5_root.is_dir():
        raise SystemExit(
            f"No folder {exp5_root}. The batch job has not created exp5 output yet "
            "(still loading the model or failed). Check the terminal running batch_mammo_correspondence."
        )

    chain_dirs = discover_chain_dirs(exp5_root)
    if args.exp5_views.strip():
        specs = batch.parse_exp5_view_specs(args.exp5_views)
        filtered: list[Path] = []
        for chain_dir in chain_dirs:
            cj = list(chain_dir.glob("*_chain.json"))
            if cj:
                d = json.loads(cj[0].read_text(encoding="utf-8"))
                key = (str(d["patient_id"]), str(d["laterality"]).upper()[:1], str(d["view"]).upper())
            else:
                m = CHAIN_DIR_RE.match(chain_dir.name)
                if not m:
                    continue
                key = (m.group("pid"), m.group("lat").upper(), m.group("view").upper())
            if key in specs:
                filtered.append(chain_dir)
        chain_dirs = filtered
    if not chain_dirs:
        raise SystemExit(
            f"No exp5 chain folders under {exp5_root}. "
            "Wait until at least one hop completes (step_* folder with correspondence or TPS meta)."
        )

    records = batch.load_roi_table(pack_root)
    rec_by_key = {(r.patient_id, r.laterality.upper()[:1], r.view.upper()): r for r in records}

    out_base = Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"NEW TPS output root: {out_base}")

    n_done = 0
    written_pngs: list[Path] = []
    for chain_dir in chain_dirs:
        chain_jsons = list(chain_dir.glob("*_chain.json"))
        if chain_jsons:
            data = json.loads(chain_jsons[0].read_text(encoding="utf-8"))
            pid = str(data["patient_id"])
            lat = str(data["laterality"]).upper()[:1]
            view = str(data["view"]).upper()
            anchor_date = data["anchor_date"]
            steps = steps_from_chain_json(chain_dir, args.exp5_max_steps)
            step_iter: list[tuple[dict, Path | None]] = [
                (s, chain_dir / f"step_{int(s['step']):02d}_{anchor_date}_to_{s['target_exam']}")
                for s in steps
            ]
        else:
            m = CHAIN_DIR_RE.match(chain_dir.name)
            if not m:
                print(f"skip {chain_dir.name}: cannot parse chain folder name")
                continue
            pid = m.group("pid")
            lat = m.group("lat").upper()
            view = m.group("view").upper()
            anchor_date = anchor_date_from_slug(m.group("anchor"))
            raw_steps = steps_from_step_dirs(
                chain_dir, pack_root, pid, lat, view, anchor_date, args.exp5_max_steps
            )
            step_iter = [(s, s.pop("_step_dir")) for s in raw_steps]

        if not step_iter:
            print(f"\n{chain_dir.name}: no completed steps yet")
            continue

        rec = rec_by_key.get((pid, lat, view))
        if rec is None:
            print(f"skip {chain_dir.name}: no roi_coords row for {pid} {lat} {view}")
            continue
        anchor_path = batch.resolve_exam_image(pack_root, pid, anchor_date, lat, view)
        if anchor_path is None:
            anchor_path = batch.resolve_clean_path(pack_root, rec.image_path)
        src_box, _, _ = batch.roi_on_record(anchor_path, rec, require_on_tissue=False)

        print(f"\n{chain_dir.name}: {len(step_iter)} hop(s), modes={','.join(modes)}")
        label = "Experiment 5: TPS only (reused SemCorre keypoints)"
        for step, step_dir in step_iter:
            step_i = int(step["step"])
            if step_dir is None or not step_dir.is_dir():
                print(f"  step {step_i}: missing folder {step_dir}")
                continue
            src_path = Path(step["source_path"])
            trg_path = Path(step["target_path"])
            sk = step["source_kp_512"]
            tk = step["target_kp_512"]
            method_src = (float(sk["x"]), float(sk["y"]))
            method_dst = (float(tk["x"]), float(tk["y"]))
            src_t = batch.load_image_chw(src_path)
            trg_t = batch.load_image_chw(trg_path)
            stem = file_stem_for_step(step_dir)
            detail = (
                f"patient {pid} | {lat} {view} | step {step_i}: "
                f"{src_path.parent.name} → {step['target_exam']} | TPS-only"
            )
            save_folder = (
                out_base / chain_dir.name / step_dir.name if out_base else step_dir
            )
            save_folder.mkdir(parents=True, exist_ok=True)
            results = tps_roi_warp.save_tps_warp_for_correspondence(
                src_display=src_t,
                trg_display=trg_t,
                method_src=method_src,
                method_dst=method_dst,
                save_folder=save_folder,
                file_stem=stem,
                src_gt_box=src_box if step_i == 1 else None,
                roi_size_ref_box=src_box,
                experiment_type=label,
                experiment_detail=detail,
                modes=modes,
            )
            for mode, res in results.items():
                written_pngs.append(res.figure_path)
                print(f"  step {step_i} [{mode}]: {res.figure_path}")
            n_done += 1

    if n_done == 0:
        raise SystemExit(
            f"No TPS regenerated under {run_root}. "
            "Ensure batch has finished at least one hop (step_* with *_correspondences_estimated.png "
            "or *_tps_*_warp_meta.json)."
        )
    print(f"\nDone. {n_done} pair(s) × modes [{args.tps_modes}] = {len(written_pngs)} new PNG(s).")
    if out_base:
        print(f"Open folder: {out_base}")
    else:
        print("Note: wrote in-place under --batch-run-dir (use --out-dir next time for a separate folder).")


if __name__ == "__main__":
    main()
