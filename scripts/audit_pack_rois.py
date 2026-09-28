#!/usr/bin/env python3
"""Check roi_coords.csv boxes against each row's image_path (roi_overlays_cancer5 pack)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from batch_mammo_correspondence import (  # noqa: E402
    load_index,
    load_roi_table,
    resolve_clean_path,
    roi_on_record,
    view_exam_timeline,
)
from utils.embed_roi import embed_box_on_png  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pack-dir",
        type=Path,
        default=ROOT.parent
        / "StableKeypointsPlus"
        / "local"
        / "data"
        / "packs"
        / "roi_overlays_cancer5",
    )
    args = p.parse_args()
    pack_root = args.pack_dir.expanduser().resolve()
    records = load_roi_table(pack_root)
    index = load_index(pack_root)

    print(f"Pack: {pack_root}")
    print(f"Rows in roi_coords (after dedupe): {len(records)}\n")

    max_center_err = 0.0
    for r in records:
        overlay = pack_root / r.image_path
        clean = resolve_clean_path(pack_root, r.image_path)
        for label, path in (("overlay", overlay), ("clean", clean)):
            if not path.is_file():
                print(f"MISSING {label}: {r.patient_id} {r.exam_date} {r.laterality} {r.view} -> {path}")
                continue
            _, center, on_tissue, _ = embed_box_on_png(
                path, r.roi_coord, reference_center=r.roi_center
            )
            cx_csv, cy_csv = r.roi_center
            err = ((center[0] - cx_csv) ** 2 + (center[1] - cy_csv) ** 2) ** 0.5
            max_center_err = max(max_center_err, err)
            flag = "OK" if err <= 2.0 else "CENTER_MISMATCH"
            tissue = "on-tissue" if on_tissue else "OFF-TISSUE"
            if flag != "OK" or not on_tissue:
                print(
                    f"{flag:16} {r.patient_id} {r.exam_date} {r.laterality:1}{r.view:3} "
                    f"{label:7} err={err:.1f}px csv=({cx_csv:.1f},{cy_csv:.1f}) "
                    f"embed=({center[0]:.1f},{center[1]:.1f}) {tissue}"
                )

        timeline = view_exam_timeline(
            pack_root, index, r.patient_id, r.laterality, r.view
        )
        newest = timeline[-1] if timeline else "?"
        if timeline and r.exam_date != newest:
            print(
                f"NOTE             {r.patient_id} {r.laterality}{r.view} ROI exam {r.exam_date} "
                f"!= newest in timeline {newest} (exp5 chains stop at ROI exam)"
            )

        box512, pt512, ok = roi_on_record(clean, r, require_on_tissue=False)
        if box512 is None:
            print(f"NO_BOX           {r.patient_id} {r.exam_date} {r.laterality} {r.view}")

    print(f"\nMax center error vs roi_center (px): {max_center_err:.2f}")
    print("Pack index join: draw roi_coord on image_path PNG (no red overlay in this pack).")


if __name__ == "__main__":
    main()
