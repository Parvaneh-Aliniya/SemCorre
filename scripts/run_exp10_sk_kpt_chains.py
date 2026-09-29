"""Exp10: SemCorre temporal transfer for each SK train keypoint (7), compare to SK on test exams.

Reads ``exp10_sk_keypoints.json`` from StableKeypointsPlus ``run_exp10_sk_semcorre_patient.py``.
For each arm (train_first → forward chain, train_last → backward), each bucket, each kpt index:
  - Anchor = train exam; propagate point hop-by-hop (transfer mode, like Exp5).
  - Compare SemCorre xy vs SK xy on every shared exam date (L2 in 512 px).

Example:

  python scripts/run_exp10_sk_kpt_chains.py \\
    --export-json /scratch/.../exp10_sk_keypoints.json \\
    --pack-dir /scratch/.../roi_overlays_exp5_temporal \\
    --out-dir /scratch/.../exp10_p62877247/semcorre_chains \\
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from batch_mammo_correspondence import (  # noqa: E402
    load_index,
    load_ldm,
    resolve_exam_image,
    view_exam_timeline,
)
from interactive_correspondence import (  # noqa: E402
    SinglePairDataset,
    load_image_chw,
    run_correspondence,
    run_roundtrip_if_needed,
)


def _hop_dates(timeline: list[str], anchor: str, direction: str) -> list[str]:
    if anchor not in timeline:
        return []
    i = timeline.index(anchor)
    if direction == "forward":
        return timeline[i + 1 :]
    return list(reversed(timeline[:i]))


def _run_kpt_chain(
    ldm,
    *,
    pack_root: Path,
    patient_id: str,
    lat: str,
    view: str,
    anchor_date: str,
    direction: str,
    src_xy: tuple[float, float],
    out_dir: Path,
    kpt_idx: int,
    device: str,
    hyper: dict,
) -> dict[str, tuple[float, float]]:
    index = load_index(pack_root)
    timeline = view_exam_timeline(pack_root, index, patient_id, lat, view)
    hops = _hop_dates(timeline, anchor_date, direction)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = resolve_exam_image(pack_root, patient_id, anchor_date, lat, view)
    if src_path is None:
        print(f"  skip kpt{kpt_idx}: no anchor image {anchor_date}", flush=True)
        return {anchor_date: src_xy}

    sem_xy: dict[str, tuple[float, float]] = {anchor_date: src_xy}
    src_t = load_image_chw(src_path)
    src_xy_cur = src_xy
    cur_path = src_path
    cur_t = src_t

    for step, trg_date in enumerate(hops, start=1):
        trg_path = resolve_exam_image(pack_root, patient_id, trg_date, lat, view)
        if trg_path is None:
            continue
        trg_t = load_image_chw(trg_path)
        step_dir = out_dir / f"hop_{step:02d}_{trg_date}"
        step_dir.mkdir(parents=True, exist_ok=True)
        stem = f"kpt{kpt_idx:02d}_step{step:02d}_{trg_date}"
        mini = next(
            iter(
                DataLoader(
                    SinglePairDataset(cur_t, trg_t, src_xy_cur),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )
        detail = (
            f"exp10 patient {patient_id} {lat}-{view} kpt {kpt_idx} "
            f"hop {step} {cur_path.parent.name} → {trg_date}"
        )
        est, *_rest = run_correspondence(
            ldm,
            mini,
            save_folder=step_dir,
            file_stem=stem,
            source_path=str(cur_path),
            target_path=str(trg_path),
            src_display=cur_t,
            trg_display=trg_t,
            src_gt_box=None,
            trg_gt_box=None,
            device=device,
            experiment_type="exp10_sk_semcorre_chain",
            experiment_detail=detail,
            **hyper,
        )
        tx, ty = float(est[0].item()), float(est[1].item())
        run_roundtrip_if_needed(
            ldm,
            src_gt_box=None,
            trg_gt_box=None,
            trg_t=trg_t,
            src_t=cur_t,
            forward_target_kp=(tx, ty),
            forward_src_kp=(float(src_xy_cur[0]), float(src_xy_cur[1])),
            original_src_path=str(cur_path),
            original_trg_path=str(trg_path),
            save_folder=step_dir,
            file_stem=stem,
            device=device,
            hyper=hyper,
            experiment_type="exp10_sk_semcorre_chain",
            experiment_detail=detail,
        )
        sem_xy[trg_date] = (tx, ty)
        src_xy_cur = (tx, ty)
        cur_path = trg_path
        cur_t = trg_t
        print(f"    kpt{kpt_idx} hop{step} → {trg_date} ({tx:.1f},{ty:.1f})", flush=True)
    return sem_xy


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-json", type=Path, required=True)
    p.add_argument("--pack-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--layers", type=int, nargs="+", default=[7, 8, 9, 10])
    p.add_argument("--noise_level", type=int, default=-8)
    p.add_argument("--num_opt_iterations", type=int, default=3)
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--limit-kpts", type=int, default=0, help="0 = all top_k")
    args = p.parse_args()

    data = json.loads(args.export_json.read_text(encoding="utf-8"))
    pack_root = args.pack_dir.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    patient_blocks: list[tuple[str, dict]] = []
    if data.get("patients"):
        for pid, block in data["patients"].items():
            patient_blocks.append((str(pid), block))
    elif data.get("patient_id"):
        patient_blocks.append((str(data["patient_id"]), {"arms": data.get("arms", {})}))
    else:
        raise SystemExit("export JSON needs 'patients' or 'patient_id'")

    want_cuda = args.device.startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required; submit on gh GPU node.")
    device = args.device if want_cuda else "cpu"
    print(f"Loading LDM on {device}...", flush=True)
    ldm = load_ldm(device, "CompVis/stable-diffusion-v1-4")
    hyper = dict(
        upsample_res=512,
        num_steps=129,
        noise_level=args.noise_level,
        layers=args.layers,
        lr=0.0023755632081200314,
        num_opt_iterations=args.num_opt_iterations,
        num_iterations=args.num_iterations,
        sigma=27.97853316316864,
        flip_prob=0.0,
        crop_percent=93.16549294381423,
        gt_mode="gaussian",
    )

    summary_rows: list[dict] = []
    for pid, block in patient_blocks:
        print(f"\n=== Exp10 SemCorre chains patient {pid} ===", flush=True)
        for arm_name, arm in block.get("arms", {}).items():
            direction = arm.get("chain_direction", "backward")
            top_k = int(arm.get("top_k", data.get("top_k", 7)))
            n_k = top_k if args.limit_kpts <= 0 else min(top_k, args.limit_kpts)
            for bucket, exams in arm.get("buckets", {}).items():
                lat, view = bucket.split("_", 1)
                by_date = {e["exam_date"]: e for e in exams}
                train_date = arm.get("train_exam_date")
                if not train_date or train_date not in by_date:
                    continue
                train_xy_list = by_date[train_date]["xy_512"]
                arm_bucket_dir = out_root / f"patient_{pid}" / arm_name / bucket
                for ki in range(n_k):
                    src = train_xy_list[ki]
                    src_xy = (float(src[0]), float(src[1]))
                    chain_dir = arm_bucket_dir / f"kpt_{ki:02d}"
                    sem = _run_kpt_chain(
                        ldm,
                        pack_root=pack_root,
                        patient_id=pid,
                        lat=lat,
                        view=view,
                        anchor_date=train_date,
                        direction=direction,
                        src_xy=src_xy,
                        out_dir=chain_dir,
                        kpt_idx=ki,
                        device=device,
                        hyper=hyper,
                    )
                    for date, sk_row in by_date.items():
                        sk_pts = sk_row["xy_512"]
                        if ki >= len(sk_pts):
                            continue
                        sk_xy = (float(sk_pts[ki][0]), float(sk_pts[ki][1]))
                        if date not in sem:
                            continue
                        d = _dist(sk_xy, sem[date])
                        summary_rows.append(
                            {
                                "patient_id": pid,
                                "arm": arm_name,
                                "bucket": bucket,
                                "kpt_index": ki,
                                "exam_date": date,
                                "sk_x": sk_xy[0],
                                "sk_y": sk_xy[1],
                                "semcorre_x": sem[date][0],
                                "semcorre_y": sem[date][1],
                                "dist_512_px": d,
                                "is_train_exam": date == train_date,
                            }
                        )

    import csv

    csv_path = out_root / "exp10_sk_vs_semcorre_distances.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    (out_root / "exp10_compare_summary.json").write_text(
        json.dumps(
            {
                "n_patients": len(patient_blocks),
                "n_rows": len(summary_rows),
                "csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {csv_path} ({len(summary_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
