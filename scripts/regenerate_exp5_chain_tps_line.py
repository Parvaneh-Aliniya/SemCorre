"""Rebuild exp5 chain overview with TPS line-align from an existing *_chain.json (no LDM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_mammo_correspondence as batch  # noqa: E402
from interactive_correspondence import (  # noqa: E402
    ChainPanelInfo,
    box_from_center_size,
    save_chain_overview_figure,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chain-json", type=str, required=True)
    p.add_argument("--pack-dir", type=str, default="../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5")
    p.add_argument("--tps-line-control", type=str, default="roi,breast", help="roi and/or breast")
    args = p.parse_args()

    chain_json = Path(args.chain_json).expanduser().resolve()
    pack_root = Path(args.pack_dir).expanduser().resolve()
    data = json.loads(chain_json.read_text(encoding="utf-8"))
    chain_dir = chain_json.parent
    pid = str(data["patient_id"])
    lat = str(data["laterality"]).upper()[:1]
    view = str(data["view"]).upper()
    anchor_date = data["anchor_date"]
    records = batch.load_roi_table(pack_root)
    rec = next((r for r in records if r.patient_id == pid and r.laterality.upper().startswith(lat) and r.view.upper() == view), None)
    if rec is None:
        raise SystemExit(f"No roi_coords row for {pid} {lat} {view}")

    steps = data.get("steps") or []
    if not steps:
        raise SystemExit("chain.json has no steps")

    anchor_path = Path(steps[0]["source_path"])
    anchor_t = batch.load_image_chw(anchor_path)
    src_box, anchor_xy, _ = batch.roi_on_record(anchor_path, rec, require_on_tissue=False)
    panels: list[ChainPanelInfo] = [
        ChainPanelInfo(
            exam_date=anchor_date,
            title=anchor_path.name,
            display=anchor_t,
            keypoint=anchor_xy,
            gt_box=src_box,
            role="anchor (ROI exam)",
            exam_year=anchor_date[:4],
            exam_name=anchor_date,
        )
    ]
    for st in steps:
        trg_path = Path(st["target_path"])
        tk = st["target_kp_512"]
        panels.append(
            ChainPanelInfo(
                exam_date=st["target_exam"],
                title=trg_path.name,
                display=batch.load_image_chw(trg_path),
                keypoint=(float(tk["x"]), float(tk["y"])),
                white_box=box_from_center_size(
                    float(tk["x"]),
                    float(tk["y"]),
                    src_box[2] - src_box[0],
                    src_box[3] - src_box[1],
                )
                if src_box
                else None,
                role=f"step {st['step']} target",
                exam_year=str(st["target_exam"])[:4],
                exam_name=str(st["target_exam"]),
                step_index=int(st["step"]),
            )
        )

    stem = chain_json.stem.replace("_chain", "")
    label = "Experiment 5: chain overview (TPS line align)"
    detail = f"patient {pid} | {lat} {view} | from {anchor_date}"
    for mode in [m.strip() for m in args.tps_line_control.split(",") if m.strip()]:
        suffix = "roi" if mode == "roi" else "breast10"
        out = chain_dir / f"{stem}_chain_overview_tps_line_{suffix}.png"
        save_chain_overview_figure(
            panels,
            out,
            experiment_type=label,
            experiment_detail=f"{detail} | TPS controls: {mode}",
            align_mode="tps_line",
            tps_line_control=mode,
            roi_size_ref_box=src_box,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
