#!/usr/bin/env python3
"""Rank smallest EMBED ROI_coords across the FULL metadata table (not just cancer5).

Vista login (tables are on scratch, not the Windows laptop):

  export EMBED_DATA_DIR=/scratch/11364/paliniya/embed_dataset/tables
  python scripts/select_smallest_embed_rois.py --top 25

Prints the smallest boxes, then one pick each for exp1 / exp2 / exp3 / exp5.
Flags whether that patient is already in the roi_overlays_cancer5 pack.
"""
from __future__ import annotations

import argparse
import ast
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

PACK_PATIENTS = {"11513410", "31292781", "45209155", "62877247", "90615275"}


def _pick(columns, *subs: str) -> str | None:
    """Prefer an exact column name, then a whole-name match, never a substring of another field."""
    lower = {c.lower(): c for c in columns}
    for s in subs:
        if s.lower() in lower:
            return lower[s.lower()]
    for s in subs:
        sl = s.lower()
        for c in columns:
            if c.lower() == sl:
                return c
    return None


def _norm_side(val) -> str:
    s = str(val or "").strip().upper()
    if s.startswith("L"):
        return "L"
    if s.startswith("R"):
        return "R"
    return "?"


def _norm_view(val) -> str:
    s = str(val or "").strip().upper()
    if "MLO" in s:
        return "MLO"
    if "CC" in s:
        return "CC"
    return "?"


def _norm_id(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def parse_boxes(raw) -> list[list[float]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        boxes = list(raw)
    else:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "[]"):
            return []
        try:
            boxes = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
    out = []
    for box in boxes if isinstance(boxes, (list, tuple)) else []:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (TypeError, ValueError):
            continue
        if ymax <= ymin or xmax <= xmin:
            continue
        out.append([ymin, xmin, ymax, xmax])
    return out


def resolve_metadata(path: Path | None) -> Path:
    if path and path.is_file():
        return path
    env = os.environ.get("EMBED_DATA_DIR", "")
    roots = [Path(env)] if env else []
    roots += [
        Path("/scratch/11364/paliniya/embed_dataset/tables"),
        Path("/scratch/11364/paliniya/EMBED/tables"),
    ]
    for root in roots:
        for name in (
            "EMBED_OpenData_metadata.csv",
            "EMBED_OpenData_metadata_reduced.csv",
        ):
            cand = root / name
            if cand.is_file():
                return cand
            cand2 = root / "tables" / name
            if cand2.is_file():
                return cand2
    raise SystemExit(
        "No EMBED metadata CSV. On Vista: export EMBED_DATA_DIR=/scratch/11364/paliniya/embed_dataset/tables"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--min-area", type=float, default=400.0, help="Ignore tiny/noise boxes")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    meta_path = resolve_metadata(args.metadata_csv)
    header = pd.read_csv(meta_path, nrows=0)
    cols = list(header.columns)
    c_pid = _pick(cols, "empi_anon")
    c_eid = _pick(cols, "acc_anon")
    c_side = _pick(cols, "ImageLateralityFinal", "ImageLaterality")
    c_view = _pick(cols, "ViewPosition")
    c_roi = _pick(cols, "ROI_coords")
    c_num = _pick(cols, "num_ROI")
    c_type = _pick(cols, "FinalImageType")
    c_date = _pick(cols, "study_date_anon")
    if not c_pid or not c_roi:
        raise SystemExit(f"Need empi_anon + ROI_coords in {meta_path}. Have: {cols[:20]}")

    usecols = [c for c in (c_pid, c_eid, c_side, c_view, c_roi, c_num, c_type, c_date) if c]
    print(f"Reading {meta_path}", flush=True)
    print(f"  columns: {usecols}", flush=True)
    df = pd.read_csv(meta_path, usecols=usecols, low_memory=False)
    print(f"  rows: {len(df):,}", flush=True)

    if c_type:
        t = df[c_type].astype(str).str.strip().str.upper()
        df = df.loc[~t.isin(("3D", "DBT", "TOMO"))].copy()
    df["patient_id"] = df[c_pid].map(_norm_id)
    df["exam_id"] = df[c_eid].map(_norm_id) if c_eid else ""
    df["laterality"] = df[c_side].map(_norm_side) if c_side else "?"
    df["view"] = df[c_view].map(_norm_view) if c_view else "?"
    if c_date:
        df["exam_date"] = df[c_date].astype(str).str.slice(0, 10)
    else:
        df["exam_date"] = df["exam_id"]

    exams_all = (
        df.groupby("patient_id")["exam_date"]
        .nunique()
        .rename("n_exams")
        .reset_index()
    )

    if c_num:
        nums = pd.to_numeric(df[c_num], errors="coerce").fillna(0)
        roi_df = df.loc[nums > 0].copy()
    else:
        roi_df = df.copy()
    roi_s = roi_df[c_roi].astype(str).str.strip()
    roi_df = roi_df.loc[roi_s.notna() & ~roi_s.isin(("", "nan", "None", "[]", "NAN"))].copy()

    rows = []
    raw_s = roi_df[c_roi].tolist()
    pids = roi_df["patient_id"].tolist()
    eids = roi_df["exam_id"].tolist()
    sides = roi_df["laterality"].tolist()
    views = roi_df["view"].tolist()
    dates = roi_df["exam_date"].tolist()
    for raw, pid, eid, side, view, date in zip(raw_s, pids, eids, sides, views, dates):
        boxes = parse_boxes(raw)
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            area = (ymax - ymin) * (xmax - xmin)
            if area < args.min_area:
                continue
            rows.append(
                {
                    "patient_id": pid,
                    "exam_id": eid,
                    "exam_date": date,
                    "laterality": side,
                    "view": view,
                    "area": area,
                    "h": ymax - ymin,
                    "w": xmax - xmin,
                    "ymin": ymin,
                    "xmin": xmin,
                    "ymax": ymax,
                    "xmax": xmax,
                    "in_cancer5_pack": pid in PACK_PATIENTS,
                }
            )

    boxes_df = pd.DataFrame(rows)
    if boxes_df.empty:
        raise SystemExit("No parsed ROI boxes.")
    boxes_df = boxes_df.merge(exams_all, on="patient_id", how="left")
    boxes_df = boxes_df.sort_values("area", kind="mergesort")

    print(f"\nParsed {len(boxes_df):,} boxes on {boxes_df['patient_id'].nunique():,} patients")
    print(f"Top {args.top} smallest (area >= {args.min_area:g}):\n")
    show = boxes_df.head(args.top)
    print(
        show[
            [
                "area",
                "patient_id",
                "exam_date",
                "laterality",
                "view",
                "w",
                "h",
                "n_exams",
                "in_cancer5_pack",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.0f}")
    )

    # --- experiment picks from the FULL ROI table ---
    by_key = defaultdict(list)
    for r in boxes_df.itertuples(index=False):
        by_key[(r.patient_id, r.exam_id, r.laterality, r.view)].append(r)

    both_view = []
    for (pid, eid, lat, view), recs in list(by_key.items()):
        if view != "CC":
            continue
        other = by_key.get((pid, eid, lat, "MLO"))
        if not other:
            continue
        a = min(x.area for x in recs + other)
        both_view.append((a, pid, eid, lat, recs[0].exam_date, recs[0].n_exams))
    both_view.sort()

    bilateral = []
    for (pid, eid, lat, view), recs in list(by_key.items()):
        if lat != "L":
            continue
        other = by_key.get((pid, eid, "R", view))
        if not other:
            continue
        a = min(x.area for x in recs + other)
        bilateral.append((a, pid, eid, view, recs[0].exam_date, recs[0].n_exams))
    bilateral.sort()

    exp5 = boxes_df.loc[boxes_df["n_exams"].fillna(0) >= 3].sort_values("area")

    print("\n=== Picks from FULL EMBED ROI_coords (not the 5-patient pack) ===")
    if both_view:
        a, pid, eid, lat, day, n = both_view[0]
        print(
            f"exp1 (smallest both-view same exam): {pid} {lat} CC+MLO  "
            f"exam {day} min_area={a:.0f} n_exams={n:.0f}  pack={pid in PACK_PATIENTS}"
        )
    if bilateral:
        a, pid, eid, view, day, n = bilateral[0]
        print(
            f"exp2 (smallest bilateral same exam+view): {pid} {view} L+R  "
            f"exam {day} min_area={a:.0f} n_exams={n:.0f}  pack={pid in PACK_PATIENTS}"
        )
    if both_view and len(both_view) >= 2:
        a1, p1, *_ = both_view[0]
        # second both-view patient with same lat as first, for exp3
        lat0 = both_view[0][3]
        mate = next((x for x in both_view[1:] if x[3] == lat0 and x[1] != p1), both_view[1])
        print(
            f"exp3 (two smallest both-view, same laterality if possible): "
            f"{mate[1]} → {p1}  areas {mate[0]:.0f} → {a1:.0f}  "
            f"pack_src={mate[1] in PACK_PATIENTS} pack_tgt={p1 in PACK_PATIENTS}"
        )
    if not exp5.empty:
        r = exp5.iloc[0]
        print(
            f"exp5 (smallest ROI, >=3 exams): {r['patient_id']} {r['laterality']} {r['view']}  "
            f"area={r.area:.0f} n_exams={r.n_exams:.0f} date={r.exam_date}  "
            f"pack={bool(r.in_cancer5_pack)}"
        )

    in_pack = int(show["in_cancer5_pack"].sum())
    print(
        f"\nOf the top {args.top} smallest boxes, {in_pack} are in roi_overlays_cancer5. "
        "If pack=False you need to pack those patients before SemCorre."
    )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        boxes_df.head(500).to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
