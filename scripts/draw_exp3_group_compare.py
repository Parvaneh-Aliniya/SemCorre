"""Exp3 review grid: one target, many sources (small / large buckets).

Two rows × N columns (one column per source patient, fixed target + view):
  Row 1 — SOURCE mammogram (GT ROI + query point)
  Row 2 — TARGET mammogram (same patient each column; pred ROI + line per transfer)

Column title = source metadata. Figure title states the shared target once.
Column footer = IoU + center distance; figure footer = column averages.

Usage:
  python scripts/draw_exp3_group_compare.py \\
    --run-root outputs/batch_experiments/vista_exp3_cross \\
    --set-json data/exp_sets/exp3_cross_patient.json \\
    --pack-dir path/to/roi_overlays_pack
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_mammo_correspondence as batch  # noqa: E402
from interactive_correspondence import (  # noqa: E402
    COLOR_FORWARD_LINE,
    COLOR_GT,
    COLOR_PRED,
    ROI_LINEWIDTH,
    _draw_roi_rect,
    box_center_xyxy,
    format_center_error_line,
    format_iou_line,
)
from regenerate_pair_figures import _infer_trg_gt  # noqa: E402

RES = 512
BANNER_H = 80
EXP_DIR = "exp3_cross_patient"


def load_display_chw(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h > RES + 20:
        img = img.crop((0, BANNER_H, w, min(BANNER_H + RES, h)))
    if img.size != (RES, RES):
        img = img.resize((RES, RES), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.5:
        arr *= 1.0 / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _load_pt(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _path_ids(path_str: str) -> tuple[str, str, str, str]:
    p = path_str.replace("\\", "/")
    pid = ""
    for part in p.split("/"):
        if part.startswith("patient_"):
            pid = part[len("patient_") :]
            break
    date, lat, view = "", "", ""
    m = re.search(r"/(\d{4}-\d{2}-\d{2})/([LR])_([A-Z]+)", p, re.I)
    if m:
        date, lat, view = m.group(1), m.group(2).upper(), m.group(3).upper()
    else:
        m2 = re.search(r"([LR])_([A-Z]+)\.png", p, re.I)
        if m2:
            lat, view = m2.group(1).upper(), m2.group(2).upper()
    return pid, date, lat, view


def pair_lookup_key(pair: dict) -> tuple:
    return (
        str(pair["source_id"]),
        str(pair["source_date"]),
        str(pair["source_laterality"]).upper()[:1],
        str(pair["source_view"]).upper(),
        str(pair["target_id"]),
        str(pair["target_date"]),
        str(pair["target_laterality"]).upper()[:1],
        str(pair["target_view"]).upper(),
    )


def pair_job_stem(pair: dict) -> str:
    """Same slug as batch_mammo_correspondence.build_exp3_from_set (max 60 chars)."""
    sl = str(pair["source_laterality"]).upper()[:1]
    sv = str(pair["source_view"]).upper()
    tl = str(pair["target_laterality"]).upper()[:1]
    tv = str(pair["target_view"]).upper()
    spec = f"{sl}_{sv}_to_{tl}_{tv}"
    raw = (
        f"src_p{pair['source_id']}_{pair['source_date']}_"
        f"trg_p{pair['target_id']}_{pair['target_date']}_{spec}"
    )
    return batch.slugify(raw)


def index_pair_dirs(run_root: Path) -> dict[str, Path]:
    """Map batch job stem (folder name) → pair output directory."""
    out: dict[str, Path] = {}
    exp_root = run_root / EXP_DIR
    if not exp_root.is_dir():
        exp_root = run_root
    for pj in exp_root.rglob("*_pair.json"):
        meta = json.loads(pj.read_text(encoding="utf-8"))
        if meta.get("experiment") != "exp3_cross_patient" and "exp3" not in str(
            meta.get("experiment", "")
        ):
            if "trg_p" not in meta.get("stem", "") and "src_p" not in meta.get("stem", ""):
                continue
        parent = pj.parent
        stem = str(meta.get("stem") or parent.name)
        out[stem] = parent
        out[parent.name] = parent
    return out


def resolve_pair_dir(pair: dict, index: dict[str, Path]) -> Path | None:
    stem = pair_job_stem(pair)
    if stem in index:
        return index[stem]
    # tolerate manual folder names
    for k, d in index.items():
        if k == stem or (len(stem) >= 48 and k.startswith(stem[:48])):
            return d
    return None


def resolve_image(path_str: str, pack_root: Path | None) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if pack_root is None:
        raise FileNotFoundError(path_str)
    return batch.resolve_clean_path(pack_root, path_str.replace("\\", "/"))


def metrics_for_pair(pair_dir: Path, stem: str) -> tuple[float | None, dict | None]:
    pt = _load_pt(pair_dir / f"{stem}_correspondence_data.pt")
    iou = pt.get("roi_iou_pred_point") if pt else None
    ce = None
    ce_path = pair_dir / f"{stem}_center_error.json"
    if ce_path.is_file():
        ce = json.loads(ce_path.read_text(encoding="utf-8")).get("vs_primary_gt")
    if iou is None:
        pj = pair_dir / f"{stem}_pair.json"
        if pj.is_file():
            iou = json.loads(pj.read_text(encoding="utf-8")).get("roi_iou_pred_point")
    return iou, ce


def _box_from_pt(pt: dict | None, key: str):
    if not pt:
        return None
    v = pt.get(key)
    if not v:
        return None
    return tuple(float(x) for x in v)


def _est_xy(pt: dict | None) -> tuple[float, float] | None:
    if not pt:
        return None
    ek = pt.get("est_keypoints")
    if ek is not None:
        t = ek if isinstance(ek, torch.Tensor) else torch.tensor(ek)
        return float(t[0, 0, 0].item()), float(t[0, 1, 0].item())
    box = pt.get("trg_pred_roi_xyxy")
    if box:
        return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    return None


def draw_cell(
    ax,
    display: torch.Tensor,
    *,
    gt_box,
    pred_box,
    src_xy: tuple[float, float] | None = None,
    est_xy: tuple[float, float] | None = None,
    show_line: bool = False,
):
    img = display.permute(1, 2, 0).detach().cpu().numpy()
    ax.imshow(img, aspect="equal")
    ax.set_xlim(0, RES)
    ax.set_ylim(RES, 0)
    if gt_box is not None:
        _draw_roi_rect(ax, gt_box, 0.0, edgecolor=COLOR_GT, linestyle="-", label="_")
        gx, gy = box_center_xyxy(gt_box)
        ax.scatter([gx], [gy], c=COLOR_GT, s=22, marker="x", zorder=7)
    if pred_box is not None:
        _draw_roi_rect(ax, pred_box, 0.0, edgecolor=COLOR_PRED, linestyle="-", label="_")
    if src_xy is not None:
        ax.scatter([src_xy[0]], [src_xy[1]], c=COLOR_FORWARD_LINE, s=40, zorder=8)
    if est_xy is not None:
        ax.scatter([est_xy[0]], [est_xy[1]], c=COLOR_FORWARD_LINE, s=40, zorder=8)
    if show_line and gt_box is not None and est_xy is not None:
        gx, gy = box_center_xyxy(gt_box)
        ax.plot([gx, est_xy[0]], [gy, est_xy[1]], color="cyan", linewidth=1.4, linestyle=":", zorder=6)
    ax.set_axis_off()


def source_column_title(pair: dict) -> str:
    return (
        f"SOURCE\n"
        f"Patient {pair['source_id']}\n"
        f"{pair['source_date']}  {pair['source_laterality']} {pair['source_view']}"
    )


def render_group_view(
    *,
    bucket: str,
    target: dict,
    pairs: list[dict],
    pair_index: dict[tuple, Path],
    pack_root: Path | None,
    out_path: Path,
    view: str,
):
    view = view.upper()
    cols = [p for p in pairs if str(p["target_view"]).upper() == view]
    cols.sort(key=lambda p: (str(p["source_id"]), str(p["source_date"])))
    if not cols:
        return False

    t0 = cols[0]
    target_line = (
        f"TARGET (shared): patient {t0['target_id']}  |  "
        f"{t0['target_date']}  {t0['target_laterality']} {t0['target_view']}"
    )

    n = len(cols)
    fig_h = 6.2 + 0.15 * n
    fig_w = max(8, 3.4 * n)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.06, hspace=0.35, wspace=0.08)

    ious: list[float] = []
    scales: list[float] = []
    raws: list[float] = []

    for j, pair in enumerate(cols):
        pair_dir = resolve_pair_dir(pair, pair_index)
        if pair_dir is None:
            print(f"  missing run for stem {pair_job_stem(pair)!r}", flush=True)
            axes[0, j].text(0.5, 0.5, "missing run", ha="center", va="center")
            axes[1, j].set_axis_off()
            axes[0, j].set_title(source_column_title(pair), fontsize=9, linespacing=1.25)
            continue

        meta = json.loads((pair_dir / f"{meta_stem(pair_dir)}_pair.json").read_text(encoding="utf-8"))
        stem = meta["stem"]
        pt = _load_pt(pair_dir / f"{stem}_correspondence_data.pt")
        iou, ce = metrics_for_pair(pair_dir, stem)
        if iou is not None:
            ious.append(float(iou))
        if ce:
            if ce.get("dist_scale_px") is not None:
                scales.append(float(ce["dist_scale_px"]))
            raw = ce.get("dist_raw_px", ce.get("dist_native_px"))
            if raw is not None:
                raws.append(float(raw))

        src_path = resolve_image(meta["src_path"], pack_root)
        trg_path = resolve_image(meta["trg_path"], pack_root)
        src_t = load_display_chw(src_path)
        trg_t = load_display_chw(trg_path)
        sx, sy = meta.get("src_xy_512") or [0, 0]
        src_xy = (float(sx), float(sy))
        est = _est_xy(pt)
        src_gt = tuple(meta["src_gt_box_512"]) if meta.get("src_gt_box_512") else None
        pred = _box_from_pt(pt, "trg_pred_roi_xyxy") or _box_from_pt(pt, "trg_pred_white_roi_xyxy")
        trg_white = _box_from_pt(pt, "trg_pred_white_roi_xyxy")
        trg_gt = _infer_trg_gt(
            meta,
            center_err=ce,
            src_gt=src_gt,
            trg_pred=pred,
            trg_white=trg_white,
        )

        draw_cell(
            axes[0, j],
            src_t,
            gt_box=src_gt,
            pred_box=None,
            src_xy=src_xy,
            show_line=False,
        )
        draw_cell(
            axes[1, j],
            trg_t,
            gt_box=trg_gt,
            pred_box=pred,
            est_xy=est,
            show_line=True,
        )
        axes[0, j].set_title(source_column_title(pair), fontsize=9, linespacing=1.25)
        foot = format_iou_line(iou) + "\n" + format_center_error_line(ce)
        axes[1, j].text(
            0.5,
            -0.06,
            foot,
            transform=axes[1, j].transAxes,
            ha="center",
            va="top",
            fontsize=8,
            linespacing=1.35,
        )

    fig.suptitle(
        f"Experiment 3 — bucket «{bucket}»\n{target_line}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.02, 0.72, "SOURCE", fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.38, "TARGET", fontsize=11, fontweight="bold", rotation=90, va="center")

    avg_parts = []
    if ious:
        avg_parts.append(f"mean IoU {sum(ious) / len(ious):.4f}  (n={len(ious)})")
    if scales:
        avg_parts.append(f"mean center dist (512 px) {sum(scales) / len(scales):.1f}")
    if raws:
        avg_parts.append(f"mean center dist (raw px) {sum(raws) / len(raws):.1f}")
    if avg_parts:
        fig.text(0.5, 0.03, "  |  ".join(avg_parts), ha="center", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


def meta_stem(pair_dir: Path) -> str:
    pjs = list(pair_dir.glob("*_pair.json"))
    if not pjs:
        raise FileNotFoundError(pair_dir)
    return json.loads(pjs[0].read_text(encoding="utf-8"))["stem"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp3 small/large source→target comparison grids")
    ap.add_argument("--run-root", type=str, required=True, help="Batch run folder (e.g. vista_exp3_cross)")
    ap.add_argument(
        "--set-json",
        type=str,
        default=str(ROOT / "data/exp_sets/exp3_cross_patient.json"),
    )
    ap.add_argument("--pack-dir", type=str, default="", help="Resolve images if paths missing")
    ap.add_argument(
        "--bucket",
        type=str,
        default="both",
        choices=("small", "large", "both"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Default: <run-root>/exp3_group_compare",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    set_json = Path(args.set_json).expanduser().resolve()
    pack_root = Path(args.pack_dir).expanduser().resolve() if args.pack_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_root / "exp3_group_compare"

    data = json.loads(set_json.read_text(encoding="utf-8"))
    groups = data.get("groups") or []
    pair_index = index_pair_dirs(run_root)
    n_dirs = len(set(pair_index.values()))
    if n_dirs == 0:
        print(f"No exp3 pair folders under {run_root}", flush=True)
        sys.exit(1)
    print(f"Indexed {n_dirs} exp3 pair folder(s)")

    for group in groups:
        bucket = str(group.get("bucket") or "")
        if args.bucket != "both" and bucket != args.bucket:
            continue
        target = group.get("target") or {}
        pairs = group.get("pairs") or []
        views = sorted({str(p["target_view"]).upper() for p in pairs})
        for view in views:
            out_path = out_dir / f"exp3_{bucket}_target_{target.get('id')}_{view}.png"
            render_group_view(
                bucket=bucket,
                target=target,
                pairs=pairs,
                pair_index=pair_index,
                pack_root=pack_root,
                out_path=out_path,
                view=view,
            )


if __name__ == "__main__":
    main()
