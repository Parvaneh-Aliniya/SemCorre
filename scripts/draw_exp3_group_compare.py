"""Exp3 review grid: cross-patient source → target (small / large buckets).

Default output: ``exp3_combined_{CC|MLO}.png`` — both targets stacked, columns = union
of all source patients; thin ROI boxes, compact layout.

Per-bucket legacy PNGs: pass ``--separate-buckets``.

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
import matplotlib.patches as patches
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
    box_center_xyxy,
    format_center_error_line,
    format_iou_line,
)
from regenerate_pair_figures import _infer_trg_gt  # noqa: E402

RES = 512
BANNER_H = 80
EXP_DIR = "exp3_cross_patient"
EXP3_ROI_LW = 1.6
EXP3_POINT_S = 7
EXP3_GT_X_S = 10
EXP3_COL_W_IN = 2.05
EXP3_ROW_H_IN = 2.15


def load_display_chw(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        img = im.convert("RGB")
    w, h = img.size
    if w <= RES + 40 and RES + 40 < h <= RES + BANNER_H + 150:
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


def _draw_roi_rect_thin(
    ax,
    box: tuple[float, float, float, float],
    x_offset: float,
    *,
    edgecolor: str,
    linestyle: str = "-",
):
    x1, y1, x2, y2 = box
    ax.add_patch(
        patches.Rectangle(
            (x1 + x_offset, y1),
            x2 - x1,
            y2 - y1,
            linewidth=EXP3_ROI_LW,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle=linestyle,
            zorder=6,
        )
    )


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


def _blank_cell(ax, message: str) -> None:
    """Placeholder when this source→target pair was not in the Exp3 batch."""
    ax.set_facecolor("black")
    ax.set_xlim(0, RES)
    ax.set_ylim(RES, 0)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=7,
        color="#aaaaaa",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.margins(0)
    ax.set_axis_off()


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
        _draw_roi_rect_thin(ax, gt_box, 0.0, edgecolor=COLOR_GT)
        gx, gy = box_center_xyxy(gt_box)
        ax.scatter([gx], [gy], c=COLOR_GT, s=EXP3_GT_X_S, marker="x", zorder=7, linewidths=0.8)
    if pred_box is not None:
        _draw_roi_rect_thin(ax, pred_box, 0.0, edgecolor=COLOR_PRED)
    if src_xy is not None:
        ax.scatter(
            [src_xy[0]],
            [src_xy[1]],
            c=COLOR_FORWARD_LINE,
            s=EXP3_POINT_S,
            zorder=8,
            linewidths=0.4,
            edgecolors="white",
        )
    if est_xy is not None:
        ax.scatter(
            [est_xy[0]],
            [est_xy[1]],
            c=COLOR_FORWARD_LINE,
            s=EXP3_POINT_S,
            zorder=8,
            linewidths=0.4,
            edgecolors="white",
        )
    if show_line and gt_box is not None and est_xy is not None:
        gx, gy = box_center_xyxy(gt_box)
        ax.plot([gx, est_xy[0]], [gy, est_xy[1]], color="cyan", linewidth=1.0, linestyle=":", zorder=6)
    ax.margins(0)
    ax.set_axis_off()


def source_column_title(pair: dict) -> str:
    return (
        f"p{pair['source_id']}  {pair['source_date']}\n"
        f"{pair['source_laterality']} {pair['source_view']}"
    )


def all_sources_for_view(groups: list[dict], view: str) -> list[dict]:
    """Unique sources (union of small + large buckets) for this target view."""
    view = view.upper()
    by_id: dict[str, dict] = {}
    for group in groups:
        for p in group.get("pairs") or []:
            if str(p["target_view"]).upper() != view:
                continue
            sid = str(p["source_id"])
            if sid not in by_id:
                by_id[sid] = p
    return [by_id[k] for k in sorted(by_id.keys(), key=lambda x: int(x) if x.isdigit() else x)]


def find_pair_in_group(group: dict, source_id: str, view: str) -> dict | None:
    view = view.upper()
    for p in group.get("pairs") or []:
        if str(p["source_id"]) != str(source_id):
            continue
        if str(p["target_view"]).upper() != view:
            continue
        return p
    return None


def _render_one_column(
    *,
    pair: dict | None,
    src_tpl: dict,
    pair_index: dict[str, Path],
    pack_root: Path | None,
    ax_src,
    ax_trg,
    ious: list[float],
    scales: list[float],
    raws: list[float],
) -> None:
    ax_src.set_title(source_column_title(src_tpl), fontsize=7, linespacing=1.2, pad=3)
    if pair is None:
        _blank_cell(ax_src, "not in batch\n(other target only)")
        _blank_cell(ax_trg, "")
        return
    pair_dir = resolve_pair_dir(pair, pair_index)
    if pair_dir is None:
        print(f"  missing run for stem {pair_job_stem(pair)!r}", flush=True)
        _blank_cell(ax_src, "missing run")
        _blank_cell(ax_trg, "")
        return

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

    draw_cell(ax_src, src_t, gt_box=src_gt, pred_box=None, src_xy=src_xy, show_line=False)
    draw_cell(ax_trg, trg_t, gt_box=trg_gt, pred_box=pred, est_xy=est, show_line=True)
    foot = format_iou_line(iou) + "\n" + format_center_error_line(ce)
    ax_trg.text(
        0.5,
        -0.02,
        foot,
        transform=ax_trg.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        linespacing=1.25,
    )


def render_combined_exp3(
    *,
    groups: list[dict],
    pair_index: dict[str, Path],
    pack_root: Path | None,
    out_path: Path,
    view: str,
) -> bool:
    """One figure: both targets (small + large), same source columns (union of all sources)."""
    view = view.upper()
    sources = all_sources_for_view(groups, view)
    if not sources or len(groups) < 1:
        return False

    n = len(sources)
    n_blocks = len(groups)
    n_rows = 2 * n_blocks
    fig_h = EXP3_ROW_H_IN * n_rows + 0.9
    fig_w = EXP3_COL_W_IN * n + 0.5
    fig, axes = plt.subplots(n_rows, n, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.03, right=0.99, hspace=0.28, wspace=0.06)

    block_ious: list[list[float]] = [[] for _ in groups]
    block_scales: list[list[float]] = [[] for _ in groups]
    block_raws: list[list[float]] = [[] for _ in groups]

    for bi, group in enumerate(groups):
        bucket = str(group.get("bucket") or bi)
        target = group.get("target") or {}
        t0 = next((p for p in group.get("pairs") or [] if str(p["target_view"]).upper() == view), None)
        if t0 is None:
            continue
        target_line = (
            f"Target «{bucket}»: p{t0['target_id']}  {t0['target_date']}  "
            f"{t0['target_laterality']} {t0['target_view']}"
        )
        row_src, row_trg = 2 * bi, 2 * bi + 1
        fig.text(
            0.008,
            1.0 - (row_src + 0.5) / n_rows * 0.86 - 0.06,
            target_line,
            fontsize=8,
            fontweight="bold",
            rotation=90,
            va="center",
        )
        for j, src_tpl in enumerate(sources):
            pair = find_pair_in_group(group, src_tpl["source_id"], view)
            _render_one_column(
                pair=pair,
                src_tpl=src_tpl,
                pair_index=pair_index,
                pack_root=pack_root,
                ax_src=axes[row_src, j],
                ax_trg=axes[row_trg, j],
                ious=block_ious[bi],
                scales=block_scales[bi],
                raws=block_raws[bi],
            )

    fig.suptitle(f"Experiment 3 — cross-patient ({view})", fontsize=11, fontweight="bold", y=0.98)
    footer_parts = []
    for bi, group in enumerate(groups):
        bucket = str(group.get("bucket") or bi)
        ious = block_ious[bi]
        scales = block_scales[bi]
        if ious:
            footer_parts.append(f"{bucket} mean IoU {sum(ious) / len(ious):.4f} (n={len(ious)})")
        if scales:
            footer_parts.append(f"{bucket} mean dist512 {sum(scales) / len(scales):.1f}px")
    if footer_parts:
        fig.text(0.5, 0.01, "  |  ".join(footer_parts), ha="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


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
    fig_h = EXP3_ROW_H_IN * 2 + 0.85
    fig_w = EXP3_COL_W_IN * n + 0.4
    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.04, right=0.99, hspace=0.25, wspace=0.06)

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
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
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
        "--separate-buckets",
        action="store_true",
        help="Also write per-bucket exp3_{small|large}_target_*.png (default: combined only)",
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

    active_groups = [
        g
        for g in groups
        if args.bucket == "both" or str(g.get("bucket") or "") == args.bucket
    ]
    all_pairs = [p for g in active_groups for p in (g.get("pairs") or [])]
    views = sorted({str(p["target_view"]).upper() for p in all_pairs})

    if len(active_groups) >= 1:
        for view in views:
            out_path = out_dir / f"exp3_combined_{view}.png"
            render_combined_exp3(
                groups=active_groups,
                pair_index=pair_index,
                pack_root=pack_root,
                out_path=out_path,
                view=view,
            )

    if args.separate_buckets:
        for group in active_groups:
            bucket = str(group.get("bucket") or "")
            target = group.get("target") or {}
            pairs = group.get("pairs") or []
            g_views = sorted({str(p["target_view"]).upper() for p in pairs})
            for view in g_views:
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
