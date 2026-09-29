"""
Interactive semantic correspondence for custom image pairs (e.g. mammograms).

Run from project root:
    python scripts/interactive_correspondence.py

Pick source/target images and output folder via file dialogs (or use CLI flags).
Click once on the source image, then enter title/description in the terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.optimize_token import (  # noqa: E402
    find_max_pixel_value,
    gaussian_circle,
    load_ldm,
    optimize_prompt,
    roi_box_mask,
    run_image_with_tokens_cropped,
    visualize_image_with_points,
)
@dataclass
class ImagePack:
    """clean: no red ROI overlay (for SD). display: original colors (for figures). roi: GT box xyxy."""

    clean: torch.Tensor
    display: torch.Tensor
    roi_box: Optional[tuple[float, float, float, float]] = None


def detect_red_roi_mask(arr_hwc: np.ndarray) -> np.ndarray:
    """Mask drawn red ROI overlays (RGB image, 0–255 or 0–1).

    Mammogram exports often use anti-aliased dark red (high R, but G/B not near zero).
    """
    if arr_hwc.max() <= 1.01:
        r = arr_hwc[..., 0] * 255.0
        g = arr_hwc[..., 1] * 255.0
        b = arr_hwc[..., 2] * 255.0
    else:
        r = arr_hwc[..., 0]
        g = arr_hwc[..., 1]
        b = arr_hwc[..., 2]
    pure_red = (r > 180) & (g < 100) & (b < 100)
    dominant_red = (r > 65) & (r > g + 12) & (r > b + 12) & ((r - np.maximum(g, b)) > 10)
    mask = pure_red | dominant_red
    # Thin box borders: dilate so sparse pixels still form one ROI
    try:
        from scipy import ndimage

        mask = ndimage.binary_dilation(mask, iterations=2)
    except ImportError:
        pass
    return mask


def detect_red_roi_box(arr_hwc: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    mask = detect_red_roi_mask(arr_hwc)
    n = int(mask.sum())
    if n < 12:
        return None
    ys, xs = np.where(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def strip_red_overlay(arr_hwc: np.ndarray) -> np.ndarray:
    mask = detect_red_roi_mask(arr_hwc)
    out = arr_hwc.copy()
    if not mask.any():
        return out
    fill = np.median(out[~mask], axis=0) if (~mask).any() else np.array([0.5, 0.5, 0.5], dtype=out.dtype)
    out[mask] = fill
    return out


def _hwc_to_chw(arr_hwc: np.ndarray) -> torch.Tensor:
    if arr_hwc.dtype != np.float32:
        arr_hwc = arr_hwc.astype(np.float32) / 255.0
    return torch.tensor(np.transpose(arr_hwc, (2, 0, 1)))


def load_image_pack(path: str) -> ImagePack:
    """Load 512 RGB; detect red GT ROI; clean copy for inference."""
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), Image.BILINEAR)
    display_hwc = np.array(image, dtype=np.float32) / 255.0
    clean_hwc = strip_red_overlay(display_hwc)
    roi = detect_red_roi_box(display_hwc)
    return ImagePack(
        clean=_hwc_to_chw(clean_hwc),
        display=_hwc_to_chw(display_hwc),
        roi_box=roi,
    )


def box_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _heatmap_gt_window_sums(m: np.ndarray, box_w: float, box_h: float) -> tuple[np.ndarray, int, int]:
    h, w = m.shape
    bw = max(1, min(w, int(round(box_w))))
    bh = max(1, min(h, int(round(box_h))))
    if bh > h or bw > w:
        empty = np.array([[0.0]])
        return empty, bw, bh

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(m, (bh, bw))
    sums = windows.sum(axis=(2, 3))
    return sums, bw, bh


def sample_heatmap_at_xy(hmap: torch.Tensor, x: float, y: float) -> float:
    """Softmax attention value at integer pixel nearest (x, y) in 512 space."""
    m = hmap.detach().cpu().float()
    if m.ndim == 3:
        m = m.squeeze(0)
    h, w = m.shape
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    return float(m[iy, ix].item())


def heatmap_max_sum_box_center(
    hmap: torch.Tensor, box_w: float, box_h: float
) -> tuple[float, float]:
    """Center of a GT-sized window where the sum of heatmap values inside is largest."""
    m = np.squeeze(hmap.detach().cpu().float().numpy())
    if m.ndim != 2:
        raise ValueError(f"expected 2D heatmap, got shape {m.shape}")
    h, w = m.shape
    sums, bw, bh = _heatmap_gt_window_sums(m, box_w, box_h)
    if sums.size == 1 and sums.flat[0] == 0.0 and (bh > h or bw > w):
        return w / 2.0, h / 2.0
    iy, ix = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return ix + bw / 2.0, iy + bh / 2.0


def heatmap_max_sum_box_center_with_peak(
    hmap: torch.Tensor, box_w: float, box_h: float
) -> tuple[float, float]:
    """Max-sum GT window among placements that contain at least one global-max pixel."""
    m = np.squeeze(hmap.detach().cpu().float().numpy())
    if m.ndim != 2:
        raise ValueError(f"expected 2D heatmap, got shape {m.shape}")
    h, w = m.shape
    sums, bw, bh = _heatmap_gt_window_sums(m, box_w, box_h)
    if sums.size == 1 and sums.flat[0] == 0.0 and (bh > h or bw > w):
        return w / 2.0, h / 2.0

    max_val = float(m.max())
    peak_ys, peak_xs = np.where(m >= max_val - 1e-9)
    valid = np.zeros(sums.shape, dtype=bool)
    for py, px in zip(peak_ys, peak_xs):
        iy_lo = max(0, int(py) - bh + 1)
        iy_hi = min(sums.shape[0] - 1, int(py))
        ix_lo = max(0, int(px) - bw + 1)
        ix_hi = min(sums.shape[1] - 1, int(px))
        valid[iy_lo : iy_hi + 1, ix_lo : ix_hi + 1] = True

    if not valid.any():
        return heatmap_max_sum_box_center(hmap, box_w, box_h)

    masked = np.where(valid, sums, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return ix + bw / 2.0, iy + bh / 2.0


def _png_wh(path: str | Path | None) -> tuple[int, int] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with Image.open(p) as im:
        return im.size


def box_center_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def center_error_metrics(
    gt_box: tuple[float, float, float, float],
    pred_xy: tuple[float, float],
    *,
    native_wh: tuple[int, int] | None = None,
    canvas: float = 512.0,
) -> dict:
    """Distance from GT ROI center to predicted point.

    ``dist_512_px`` is on the SemCorre canvas. Native pixels use the real PNG
    width/height (anisotropic stretch). Percents are relative to that scale.
    """
    gx, gy = box_center_xyxy(gt_box)
    px, py = float(pred_xy[0]), float(pred_xy[1])
    dx, dy = px - gx, py - gy
    dist = math.hypot(dx, dy)
    diag = math.hypot(canvas, canvas)
    out: dict = {
        "gt_center_512": {"x": gx, "y": gy},
        "pred_center_512": {"x": px, "y": py},
        "dx_512": dx,
        "dy_512": dy,
        "dist_512_px": dist,
        "dist_scale_px": dist,
        "dist_pct_of_width": 100.0 * dist / canvas,
        "dist_pct_of_height": 100.0 * dist / canvas,
        "dist_pct_of_diagonal": 100.0 * dist / diag,
        "canvas_px": canvas,
    }
    if native_wh and native_wh[0] > 0 and native_wh[1] > 0:
        nw, nh = float(native_wh[0]), float(native_wh[1])
        dx_n, dy_n = dx * nw / canvas, dy * nh / canvas
        dist_n = math.hypot(dx_n, dy_n)
        out["native_wh"] = {"w": int(native_wh[0]), "h": int(native_wh[1])}
        out["dx_native_px"] = dx_n
        out["dy_native_px"] = dy_n
        out["dist_native_px"] = dist_n
        out["dist_raw_px"] = dist_n
        out["dist_pct_of_native_width"] = 100.0 * dist_n / nw
        out["dist_pct_of_native_height"] = 100.0 * dist_n / nh
        out["dist_pct_of_native_diagonal"] = 100.0 * dist_n / math.hypot(nw, nh)
    return out


def nearest_center_error(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_xy: tuple[float, float],
    *,
    native_wh: tuple[int, int] | None = None,
    canvas: float = 512.0,
) -> dict | None:
    best = None
    for box in gt_boxes:
        m = center_error_metrics(box, pred_xy, native_wh=native_wh, canvas=canvas)
        if best is None or m["dist_512_px"] < best["dist_512_px"]:
            best = m
    return best


COLOR_GT = "#39FF14"
COLOR_PRED = "#00E5FF"
COLOR_BACK_LINE = "#FF1493"
COLOR_FORWARD_LINE = "#FF8C00"
ROI_LINEWIDTH = 3.5
CORRESPONDENCE_ARROW_LW = 2.25
CORRESPONDENCE_ARROW_HEAD = 14


def _add_direction_arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str,
    linewidth: float = CORRESPONDENCE_ARROW_LW,
    zorder: int = 4,
    head_scale: float = CORRESPONDENCE_ARROW_HEAD,
) -> None:
    """Arrow from (x0,y0) to (x1,y1); tip ends on the target keypoint."""
    if math.hypot(x1 - x0, y1 - y0) < 1e-3:
        return
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=linewidth,
            mutation_scale=head_scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def _arrow_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLOR_FORWARD_LINE,
            lw=CORRESPONDENCE_ARROW_LW,
            marker=">",
            markersize=8,
            label="Forward",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_BACK_LINE,
            lw=CORRESPONDENCE_ARROW_LW,
            marker=">",
            markersize=8,
            label="Back",
        ),
    ]


def format_iou_line(iou: float | None) -> str:
    if iou is None:
        return "IoU pred vs GT —"
    return f"IoU pred vs GT {iou:.4f}"


def format_panel_metrics(*, iou: float | None = None, center_err: dict | None = None) -> str:
    parts = [format_iou_line(iou), format_center_error_line(center_err)]
    return "\n".join(p for p in parts if p)


def _draw_metric_banner(ax, x_offset: float, text: str) -> None:
    if not text or not text.strip():
        return
    ax.text(
        x_offset + 8,
        18,
        text,
        color="white",
        fontsize=10,
        fontweight="bold",
        va="top",
        zorder=9,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3, "edgecolor": "none"},
    )


def format_center_error_line(err: dict | None) -> str:
    """GT-center to pred-center: raw = native PNG px, scale = 512 canvas px."""
    if not err:
        return "Center dist  raw —  |  scale —"
    scale = err.get("dist_scale_px", err.get("dist_512_px"))
    raw = err.get("dist_raw_px", err.get("dist_native_px"))
    pct = err.get("dist_pct_of_width")
    line = f"Center dist  scale {scale:.1f} px"
    if pct is not None:
        line += f" ({pct:.2f}% of 512)"
    if raw is not None:
        line = f"Center dist  raw {raw:.1f} px  |  scale {scale:.1f} px"
        if pct is not None:
            line += f" ({pct:.2f}% of 512)"
    return line


def box_from_center_size(cx: float, cy: float, w: float, h: float, size: int = 512) -> tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    x1 = max(0.0, min(x1, size - 1))
    y1 = max(0.0, min(y1, size - 1))
    x2 = max(0.0, min(x2, size - 1))
    y2 = max(0.0, min(y2, size - 1))
    return x1, y1, x2, y2


def _meta_from_image_path(path: str) -> dict[str, str]:
    p = Path(path)
    laterality, view = "", ""
    stem = p.stem
    m = re.match(r"^([LR])_([A-Z]+)", stem.upper())
    if m:
        laterality, view = m.group(1), m.group(2)
        if view.endswith("_2") or view.endswith("_3"):
            view = view.rsplit("_", 1)[0]
    date = ""
    parent = p.parent.name
    if len(parent) >= 10 and parent[4] == "-" and parent[7] == "-":
        date = parent[:10]
    patient = ""
    for part in p.parts:
        if part.startswith("patient_"):
            patient = part[len("patient_") :]
            break
    return {"patient": patient, "date": date, "laterality": laterality, "view": view}


def _fmt_box_xyxy(box: Optional[tuple[float, float, float, float]]) -> str:
    if box is None:
        return "ROI —"
    x1, y1, x2, y2 = box
    return f"ROI [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"


def _fmt_point(kp) -> str:
    if kp is None:
        return "Point —"
    if hasattr(kp, "__len__"):
        return f"Point ({float(kp[0]):.0f}, {float(kp[1]):.0f})"
    return "Point —"


def format_panel_header(
    *,
    role: str,
    path: str,
    box: Optional[tuple[float, float, float, float]] = None,
    kp=None,
    exam_id: str = "",
    extra: str = "",
) -> str:
    meta = _meta_from_image_path(path)
    pid = meta["patient"] or "—"
    date = meta["date"] or "—"
    lat = meta["laterality"] or "—"
    view = meta["view"] or "—"
    eid = exam_id or "—"
    lines = [
        role,
        f"Patient {pid}",
        f"Exam {eid}   {date}",
        f"{lat} {view}",
        _fmt_box_xyxy(box),
        _fmt_point(kp),
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _boxes_close(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    tol: float = 1.5,
) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _draw_gt_roi_list(
    ax,
    boxes: list[tuple[float, float, float, float]] | None,
    x_offset: float,
    *,
    primary: tuple[float, float, float, float] | None = None,
):
    drawn = 0
    for box in boxes or []:
        if primary is not None and _boxes_close(box, primary):
            continue
        _draw_roi_rect(
            ax,
            box,
            x_offset,
            edgecolor=COLOR_GT,
            linestyle="-",
            label="GT ROI" if drawn == 0 and primary is None else "_nolegend_",
        )
        drawn += 1


def _draw_roi_rect(
    ax,
    box: tuple[float, float, float, float],
    x_offset: float,
    *,
    edgecolor: str,
    linestyle: str,
    label: str,
):
    x1, y1, x2, y2 = box
    ax.add_patch(
        patches.Rectangle(
            (x1 + x_offset, y1),
            x2 - x1,
            y2 - y1,
            linewidth=ROI_LINEWIDTH,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle=linestyle,
            label=label,
            zorder=6,
        )
    )


def save_correspondence_figure(
    src_display: torch.Tensor,
    trg_display: torch.Tensor,
    src_kp: torch.Tensor,
    est_kp: torch.Tensor,
    *,
    source_name: str,
    target_name: str,
    save_path: Path,
    src_gt_box: Optional[tuple[float, float, float, float]] = None,
    trg_gt_box: Optional[tuple[float, float, float, float]],
    src_all_gt_boxes: Optional[list] = None,
    trg_all_gt_boxes: Optional[list] = None,
    trg_pred_box: Optional[tuple[float, float, float, float]],
    trg_heatmap_box: Optional[tuple[float, float, float, float]],
    trg_peak_sum_box: Optional[tuple[float, float, float, float]] = None,
    trg_pred_white_box: Optional[tuple[float, float, float, float]] = None,
    roi_iou_pred: Optional[float] = None,
    roi_iou_heatmap: Optional[float] = None,
    roi_iou_peak_sum: Optional[float] = None,
    line_width: float = CORRESPONDENCE_ARROW_LW,
    experiment_type: str = "",
    experiment_detail: str = "",
    src_exam_id: str = "",
    trg_exam_id: str = "",
    center_error: Optional[dict] = None,
    show_trg_gt: bool = True,
    show_src_gt: bool = True,
    roundtrip_src_kp: Optional[tuple[float, float]] = None,
):
    """Side-by-side: green GT, blue pred. ``show_trg_gt=False`` hides target GT only."""
    display = torch.cat([src_display, trg_display], dim=2).permute(1, 2, 0).detach().cpu().numpy()
    sx, sy = src_kp[0].item(), src_kp[1].item()
    tx, ty = est_kp[0].item(), est_kp[1].item()
    x_off = 512.0
    pred_draw = trg_pred_box if trg_pred_box is not None else trg_pred_white_box
    src_header_box = src_gt_box or (src_all_gt_boxes[0] if src_all_gt_boxes else None)
    if show_trg_gt:
        trg_header_box = trg_gt_box or (trg_all_gt_boxes[0] if trg_all_gt_boxes else None)
    else:
        trg_header_box = pred_draw
    trg_center_err = center_error if show_trg_gt else None
    src_header = format_panel_header(
        role="SOURCE",
        path=source_name,
        box=src_header_box,
        kp=(sx, sy),
        exam_id=src_exam_id,
    )
    trg_header = format_panel_header(
        role="TARGET",
        path=target_name,
        box=trg_header_box,
        kp=(tx, ty),
        exam_id=trg_exam_id,
        extra=format_panel_metrics(iou=roi_iou_pred, center_err=trg_center_err),
    )

    fig = plt.figure(figsize=(20, 13))
    if experiment_type:
        fig.text(0.5, 0.99, experiment_type, ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.26, 0.945, src_header, ha="center", va="top", fontsize=11, linespacing=1.45)
    fig.text(0.74, 0.945, trg_header, ha="center", va="top", fontsize=11, linespacing=1.45)

    ax = fig.add_axes([0.02, 0.06, 0.96, 0.70])
    ax.imshow(display, aspect="equal")
    ax.set_xlim(0, 1024)
    ax.set_ylim(512, 0)
    _add_direction_arrow(
        ax, sx, sy, tx + x_off, ty, color=COLOR_FORWARD_LINE, linewidth=line_width, zorder=4
    )
    scatter_x = [sx, tx + x_off]
    scatter_y = [sy, ty]
    scatter_c = [COLOR_FORWARD_LINE, COLOR_FORWARD_LINE]
    if roundtrip_src_kp is not None:
        bx, by = roundtrip_src_kp
        _add_direction_arrow(
            ax, tx + x_off, ty, bx, by, color=COLOR_BACK_LINE, linewidth=line_width, zorder=4
        )
        scatter_x.append(bx)
        scatter_y.append(by)
        scatter_c.append(COLOR_BACK_LINE)
    ax.scatter(scatter_x, scatter_y, c=scatter_c, s=40, zorder=5)

    src_boxes: list = []
    if show_src_gt:
        src_boxes = list(src_all_gt_boxes or [])
        if src_gt_box is not None and not any(_boxes_close(src_gt_box, b) for b in src_boxes):
            src_boxes.insert(0, src_gt_box)
    trg_boxes: list = []
    if show_trg_gt:
        trg_boxes = list(trg_all_gt_boxes or [])
        if trg_gt_box is not None and not any(_boxes_close(trg_gt_box, b) for b in trg_boxes):
            trg_boxes.insert(0, trg_gt_box)

    if show_src_gt and src_gt_box is not None:
        _draw_roi_rect(
            ax, src_gt_box, 0.0, edgecolor=COLOR_GT, linestyle="-", label="GT ROI"
        )
    if show_src_gt and src_boxes:
        _draw_gt_roi_list(ax, src_boxes, 0.0, primary=src_gt_box)
        bx0 = src_gt_box or src_boxes[0]
        ax.scatter(
            [0.5 * (bx0[0] + bx0[2])],
            [0.5 * (bx0[1] + bx0[3])],
            c=COLOR_GT,
            s=28,
            marker="x",
            zorder=7,
        )

    if trg_boxes:
        _draw_gt_roi_list(ax, trg_boxes, x_off)
        tb = trg_gt_box or trg_boxes[0]
        ax.scatter(
            [0.5 * (tb[0] + tb[2]) + x_off],
            [0.5 * (tb[1] + tb[3])],
            c=COLOR_GT,
            s=28,
            marker="x",
            zorder=7,
        )
    if pred_draw is not None:
        _draw_roi_rect(ax, pred_draw, x_off, edgecolor=COLOR_PRED, linestyle="-", label="Pred ROI")
    if show_trg_gt and trg_gt_box is not None and center_error:
        gx, gy = box_center_xyxy(trg_gt_box)
        ax.plot(
            [gx + x_off, tx + x_off],
            [gy, ty],
            color="cyan",
            linewidth=1.6,
            linestyle=":",
            zorder=8,
            label="Center dist",
        )
        _draw_metric_banner(
            ax,
            x_off,
            format_panel_metrics(iou=roi_iou_pred, center_err=center_error),
        )

    ax.set_axis_off()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9, frameon=False)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roundtrip_source_overlay_figure(
    src_display: torch.Tensor,
    *,
    src_gt_box: tuple[float, float, float, float],
    back_pred_box: Optional[tuple[float, float, float, float]],
    back_heatmap_box: Optional[tuple[float, float, float, float]],
    back_peak_sum_box: Optional[tuple[float, float, float, float]],
    back_kp: tuple[float, float],
    roi_iou_pred: Optional[float],
    roi_iou_heatmap: Optional[float],
    roi_iou_peak_sum: Optional[float],
    source_name: str,
    save_path: Path,
    experiment_type: str = "",
    experiment_detail: str = "",
    center_error: Optional[dict] = None,
):
    """Original source image: GT ROI vs round-trip prediction overlaid."""
    display = src_display.permute(1, 2, 0).detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 13))
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.66])
    ax.imshow(display, aspect="equal")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    _draw_roi_rect(ax, src_gt_box, 0.0, edgecolor=COLOR_GT, linestyle="-", label="Original source ROI")
    bx, by = back_kp
    ax.scatter([bx], [by], c=COLOR_FORWARD_LINE, s=45, zorder=5, label="Round-trip point")
    if back_pred_box is not None:
        _draw_roi_rect(
            ax, back_pred_box, 0.0, edgecolor=COLOR_PRED, linestyle="-", label="Back-pred ROI"
        )
    ax.set_axis_off()
    header = format_panel_header(
        role="SOURCE (round-trip)",
        path=source_name,
        box=src_gt_box,
        kp=back_kp,
        extra=format_panel_metrics(iou=roi_iou_pred, center_err=center_error),
    )
    if experiment_type:
        fig.text(0.5, 0.99, experiment_type, ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, header, ha="center", va="top", fontsize=11, linespacing=1.45)
    if src_gt_box is not None and center_error:
        gx, gy = box_center_xyxy(src_gt_box)
        ax.plot([gx, bx], [gy, by], color="cyan", linewidth=1.6, linestyle=":", zorder=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9, frameon=False)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_bidirectional_pair_figure(
    src_display: torch.Tensor,
    trg_display: torch.Tensor,
    *,
    forward_src_kp: tuple[float, float],
    forward_trg_kp: tuple[float, float],
    back_src_kp: tuple[float, float],
    source_name: str,
    target_name: str,
    save_path: Path,
    src_gt_box: Optional[tuple[float, float, float, float]] = None,
    src_all_gt_boxes: Optional[list] = None,
    trg_all_gt_boxes: Optional[list] = None,
    back_pred_box: Optional[tuple[float, float, float, float]] = None,
    back_heatmap_box: Optional[tuple[float, float, float, float]] = None,
    back_peak_sum_box: Optional[tuple[float, float, float, float]] = None,
    forward_trg_pred_box: Optional[tuple[float, float, float, float]] = None,
    trg_pred_white_box: Optional[tuple[float, float, float, float]] = None,
    trg_gt_box: Optional[tuple[float, float, float, float]] = None,
    forward_roi_iou: Optional[float] = None,
    back_roi_iou: Optional[float] = None,
    forward_center_error: Optional[dict] = None,
    back_center_error: Optional[dict] = None,
    roi_iou_pred: Optional[float] = None,
    roi_iou_heatmap: Optional[float] = None,
    roi_iou_peak_sum: Optional[float] = None,
    experiment_type: str = "",
    experiment_detail: str = "",
) -> None:
    """One row: source | target — forward (orange) + back (hot pink) with GT/pred ROIs."""
    if back_roi_iou is None:
        back_roi_iou = roi_iou_pred

    display = torch.cat([src_display, trg_display], dim=2).permute(1, 2, 0).detach().cpu().numpy()
    sx, sy = forward_src_kp
    tx, ty = forward_trg_kp
    bx, by = back_src_kp
    x_off = 512.0
    trg_primary_gt = trg_gt_box or (trg_all_gt_boxes[0] if trg_all_gt_boxes else None)

    fig = plt.figure(figsize=(20, 13))
    fig.text(
        0.26,
        0.945,
        format_panel_header(
            role="SOURCE",
            path=source_name,
            box=src_gt_box,
            kp=forward_src_kp,
            extra=format_panel_metrics(iou=back_roi_iou, center_err=back_center_error),
        ),
        ha="center",
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    fig.text(
        0.74,
        0.945,
        format_panel_header(
            role="TARGET",
            path=target_name,
            box=trg_primary_gt or trg_pred_white_box,
            kp=forward_trg_kp,
            extra=format_panel_metrics(iou=forward_roi_iou, center_err=forward_center_error),
        ),
        ha="center",
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    if experiment_type:
        fig.text(0.5, 0.99, experiment_type, ha="center", va="top", fontsize=13, fontweight="bold")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.70])
    ax.imshow(display, aspect="equal")
    ax.set_xlim(0, 1024)
    ax.set_ylim(512, 0)

    _add_direction_arrow(ax, sx, sy, tx + x_off, ty, color=COLOR_FORWARD_LINE, zorder=4)
    _add_direction_arrow(ax, tx + x_off, ty, bx, by, color=COLOR_BACK_LINE, zorder=4)
    ax.scatter(
        [sx, tx + x_off, bx],
        [sy, ty, by],
        c=[COLOR_FORWARD_LINE, COLOR_FORWARD_LINE, COLOR_BACK_LINE],
        s=45,
        zorder=5,
    )

    src_boxes = list(src_all_gt_boxes or [])
    if src_gt_box is not None and not any(_boxes_close(src_gt_box, b) for b in src_boxes):
        src_boxes.insert(0, src_gt_box)
    if src_boxes:
        _draw_gt_roi_list(ax, src_boxes, 0.0)
    if src_gt_box is not None:
        gx, gy = box_center_xyxy(src_gt_box)
        ax.scatter([gx], [gy], c=COLOR_GT, s=28, marker="x", zorder=7)
    trg_boxes = list(trg_all_gt_boxes or [])
    if trg_primary_gt is not None and not any(_boxes_close(trg_primary_gt, b) for b in trg_boxes):
        trg_boxes.insert(0, trg_primary_gt)
    if trg_boxes:
        _draw_gt_roi_list(ax, trg_boxes, x_off)
    if trg_primary_gt is not None:
        gx, gy = box_center_xyxy(trg_primary_gt)
        ax.scatter([gx + x_off], [gy], c=COLOR_GT, s=28, marker="x", zorder=7)
    if back_pred_box is not None:
        _draw_roi_rect(
            ax, back_pred_box, 0.0, edgecolor=COLOR_PRED, linestyle="-", label="Back pred ROI"
        )
    trg_pred_draw = forward_trg_pred_box if forward_trg_pred_box is not None else trg_pred_white_box
    if trg_pred_draw is not None:
        _draw_roi_rect(ax, trg_pred_draw, x_off, edgecolor=COLOR_PRED, linestyle="-", label="Pred ROI")

    if src_gt_box is not None and back_center_error:
        gx, gy = box_center_xyxy(src_gt_box)
        ax.plot([gx, bx], [gy, by], color="cyan", linewidth=1.6, linestyle=":", zorder=8)
    if trg_primary_gt is not None and forward_center_error:
        gx, gy = box_center_xyxy(trg_primary_gt)
        ax.plot([gx + x_off, tx + x_off], [gy, ty], color="cyan", linewidth=1.6, linestyle=":", zorder=8)

    ax.set_axis_off()
    roi_handles, roi_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=_arrow_legend_handles() + roi_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_with_rois(
    mean_trg_map: torch.Tensor,
    save_path: Path,
    trg_gt_box: Optional[tuple[float, float, float, float]],
    trg_pred_box: Optional[tuple[float, float, float, float]],
    trg_heatmap_box: Optional[tuple[float, float, float, float]],
    trg_peak_sum_box: Optional[tuple[float, float, float, float]] = None,
    trg_pred_white_box: Optional[tuple[float, float, float, float]] = None,
    trg_display: Optional[torch.Tensor] = None,
):
    """Mean target attention heatmap with ROIs; optional image underlay."""
    m = mean_trg_map.detach().cpu().float().numpy()
    vmin, vmax = float(m.min()), float(m.max())
    fig, ax = plt.subplots(figsize=(9, 8))
    if trg_display is not None:
        base = trg_display.permute(1, 2, 0).detach().cpu().numpy()
        if base.ndim == 3 and base.shape[2] == 3:
            ax.imshow(np.clip(base, 0.0, 1.0), aspect="equal")
        else:
            ax.imshow(np.squeeze(base), cmap="gray", aspect="equal")
        heat_im = ax.imshow(m, cmap="viridis", alpha=0.55, aspect="equal", vmin=vmin, vmax=vmax)
    else:
        heat_im = ax.imshow(m, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(heat_im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Attention (softmax)  min={vmin:.4g}  max={vmax:.4g}", fontsize=10)
    for box, color, ls in (
        (trg_gt_box, "green", "-"),
        (trg_pred_box, "blue", "-"),
        (trg_pred_white_box, "blue", "-"),
    ):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none", linestyle=ls
            )
        )
    ax.set_axis_off()
    ax.set_title(f"Target attention (mean)  [{vmin:.4g}, {vmax:.4g}]")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ChainPanelInfo:
    exam_date: str
    title: str
    display: torch.Tensor
    keypoint: Optional[tuple[float, float]] = None
    gt_box: Optional[tuple[float, float, float, float]] = None
    pred_box: Optional[tuple[float, float, float, float]] = None
    heatmap_box: Optional[tuple[float, float, float, float]] = None
    peak_box: Optional[tuple[float, float, float, float]] = None
    white_box: Optional[tuple[float, float, float, float]] = None
    role: str = ""
    exam_year: str = ""
    exam_name: str = ""
    view_label: str = ""
    exam_id: str = ""
    source_keypoint: Optional[tuple[float, float]] = None
    step_index: int = 0
    heatmap_score_at_kp: Optional[float] = None
    heatmap_map_max: Optional[float] = None


def _shift_box_xyxy(
    box: tuple[float, float, float, float], dx: float, dy: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _shift_display_chw(display: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    from scipy.ndimage import shift

    planes = []
    for ch in range(display.shape[0]):
        planes.append(
            shift(
                display[ch].detach().cpu().numpy(),
                (dy, dx),
                order=1,
                mode="constant",
                cval=0.0,
            )
        )
    return torch.from_numpy(np.stack(planes, axis=0)).to(dtype=display.dtype)


def _align_chain_panel_to_ref(
    pan: ChainPanelInfo,
    ref: tuple[float, float],
) -> ChainPanelInfo:
    """Translate image/boxes so the panel keypoint sits at ref (512 px space)."""
    ref_x, ref_y = ref
    if pan.keypoint is not None:
        kx, ky = pan.keypoint
    elif pan.gt_box is not None:
        gx1, gy1, gx2, gy2 = pan.gt_box
        kx, ky = 0.5 * (gx1 + gx2), 0.5 * (gy1 + gy2)
    else:
        kx, ky = ref_x, ref_y
    dx, dy = ref_x - kx, ref_y - ky
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return replace(pan, keypoint=(ref_x, ref_y))

    disp = _shift_display_chw(pan.display, dx, dy)
    gt = _shift_box_xyxy(pan.gt_box, dx, dy) if pan.gt_box else None
    pred = _shift_box_xyxy(pan.pred_box, dx, dy) if pan.pred_box else None
    hm = _shift_box_xyxy(pan.heatmap_box, dx, dy) if pan.heatmap_box else None
    pk = _shift_box_xyxy(pan.peak_box, dx, dy) if pan.peak_box else None
    wh = _shift_box_xyxy(pan.white_box, dx, dy) if pan.white_box else None
    src_kp = None
    if pan.source_keypoint is not None:
        sx, sy = pan.source_keypoint
        src_kp = (sx + dx, sy + dy)
    return replace(
        pan,
        display=disp,
        keypoint=(ref_x, ref_y),
        gt_box=gt,
        pred_box=pred,
        heatmap_box=hm,
        peak_box=pk,
        white_box=wh,
        source_keypoint=src_kp,
    )


def _tps_align_chain_panel_to_ref(
    pan: ChainPanelInfo,
    ref: tuple[float, float],
    *,
    control_mode: str = "roi",
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
) -> ChainPanelInfo:
    """TPS-warp panel so keypoint lands on ref (horizontal chain line), not a simple shift."""
    from utils.tps_roi_warp import (
        box_same_size_at,
        rgb01_to_tensor_chw,
        tensor_chw_to_rgb01,
        tps_align_image_keypoint_to_ref,
        warp_box_xyxy_with_tps,
        warp_xy,
    )

    ref_x, ref_y = ref
    if pan.keypoint is None:
        return pan
    kx, ky = pan.keypoint
    ref_box = roi_size_ref_box or pan.gt_box
    box = pan.gt_box if pan.gt_box is not None else box_same_size_at(ref_box, (kx, ky))
    src_rgb = tensor_chw_to_rgb01(pan.display)
    warped_rgb, _warped_kp, tps = tps_align_image_keypoint_to_ref(
        src_rgb,
        (kx, ky),
        ref,
        control_mode=control_mode,  # type: ignore[arg-type]
        roi_box=box,
    )
    disp = rgb01_to_tensor_chw(warped_rgb)
    gt = warp_box_xyxy_with_tps(tps, pan.gt_box) if pan.gt_box else None
    pred = warp_box_xyxy_with_tps(tps, pan.pred_box) if pan.pred_box else None
    hm = warp_box_xyxy_with_tps(tps, pan.heatmap_box) if pan.heatmap_box else None
    pk = warp_box_xyxy_with_tps(tps, pan.peak_box) if pan.peak_box else None
    wh = warp_box_xyxy_with_tps(tps, pan.white_box) if pan.white_box else None
    src_kp = warp_xy(tps, pan.source_keypoint) if pan.source_keypoint is not None else None
    return replace(
        pan,
        display=disp,
        keypoint=(ref_x, ref_y),
        gt_box=gt,
        pred_box=pred,
        heatmap_box=hm,
        peak_box=pk,
        white_box=wh,
        source_keypoint=src_kp,
    )


def _prepare_chain_panels_for_row(
    panels: list[ChainPanelInfo],
    align_mode: str,
    align_keypoint: tuple[float, float],
    *,
    tps_line_control: str = "roi",
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
) -> list[ChainPanelInfo]:
    if align_mode == "raw":
        return panels
    if align_mode == "tps_line":
        return [
            _tps_align_chain_panel_to_ref(
                p,
                align_keypoint,
                control_mode=tps_line_control,
                roi_size_ref_box=roi_size_ref_box,
            )
            for p in panels
        ]
    return [_align_chain_panel_to_ref(p, align_keypoint) for p in panels]


def _kp_xy_for_display(
    pan: ChainPanelInfo,
    align_mode: str,
    align_keypoint: tuple[float, float],
) -> Optional[tuple[float, float]]:
    if pan.keypoint is None:
        return None
    if align_mode == "raw":
        return pan.keypoint
    return align_keypoint


def _render_chain_row_on_ax(
    ax,
    panels: list[ChainPanelInfo],
    *,
    align_mode: str = "raw",
    align_keypoint: tuple[float, float] = (256.0, 256.0),
    row_title: str = "",
    tps_line_control: str = "roi",
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
) -> None:
    """One row of original mammograms. Each pair keeps its own predicted (x, y)."""
    n = len(panels)
    if n == 0:
        return
    res = 512.0
    x_off = res
    label_band = 92.0
    draw_panels = list(panels)
    row = torch.cat([p.display for p in draw_panels], dim=2).permute(1, 2, 0).detach().cpu().numpy()
    ax.imshow(row, aspect="equal", interpolation="bilinear", extent=(0, res * n, res, 0))
    ax.set_xlim(0, res * n)
    ax.set_ylim(res + label_band, 0)
    ax.axhline(res, color="0.35", linewidth=0.8, zorder=2)
    if row_title:
        ax.set_title(row_title, fontsize=12, fontweight="bold", pad=8)

    for i in range(1, n):
        ax.axvline(i * x_off, color="white", linewidth=1.0, alpha=0.85, zorder=3)

    poly_x: list[float] = []
    poly_y: list[float] = []
    for i, pan in enumerate(draw_panels):
        ox = i * x_off
        if pan.gt_box is not None:
            _draw_roi_rect(ax, pan.gt_box, ox, edgecolor=COLOR_GT, linestyle="-", label="GT ROI")
        pred_b = pan.pred_box if pan.pred_box is not None else pan.white_box
        if pred_b is not None:
            _draw_roi_rect(ax, pred_b, ox, edgecolor=COLOR_PRED, linestyle="-", label="Pred ROI")
        if pan.gt_box is not None and pred_b is not None:
            gx, gy = box_center_xyxy(pan.gt_box)
            px, py = box_center_xyxy(pred_b)
            ax.plot(
                [gx + ox, px + ox],
                [gy, py],
                color="cyan",
                linewidth=1.4,
                linestyle=":",
                zorder=8,
            )
        kp = pan.keypoint
        if kp is not None:
            ax.scatter([kp[0] + ox], [kp[1]], c="orange", s=55, edgecolors="black", linewidths=0.8, zorder=8)
            poly_x.append(kp[0] + ox)
            poly_y.append(kp[1])
        cx = ox + res / 2
        year = (pan.exam_year or (pan.exam_date[:4] if pan.exam_date else "")).strip()
        exam_name = (pan.exam_date or pan.exam_name or "").strip()
        view = (pan.view_label or "").strip()
        role = "anchor" if pan.step_index <= 0 else f"step {pan.step_index} target"
        box = pan.gt_box or pan.white_box or pan.pred_box
        box_s = _fmt_box_xyxy(box)
        ax.text(cx, res + 6, f"{year or exam_name}  {view}".strip(), ha="center", va="top", fontsize=10, fontweight="bold", color="0.1", zorder=9)
        ax.text(cx, res + 26, f"{exam_name}  {role}".strip(), ha="center", va="top", fontsize=8, color="0.25", zorder=9)
        ax.text(cx, res + 46, box_s, ha="center", va="top", fontsize=7, color="0.35", zorder=9)

    if len(poly_x) >= 2:
        ax.plot(poly_x, poly_y, color="orange", linewidth=2.5, zorder=7, solid_capstyle="round")
    for i in range(n - 1):
        p1 = draw_panels[i + 1]
        mid_x = (i + 0.5) * x_off
        ax.text(
            mid_x,
            res + 68,
            f"pair {p1.step_index}" if p1.step_index > 0 else "",
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
            fontweight="bold",
            zorder=9,
        )
    ax.set_axis_off()


def save_chain_overview_figure(
    panels: list[ChainPanelInfo],
    save_path: Path,
    *,
    experiment_type: str,
    experiment_detail: str,
    step_lines: Optional[list[str]] = None,
    align_keypoint: tuple[float, float] = (256.0, 256.0),
    align_mode: str = "raw",
    tps_line_control: str = "roi",
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
):
    """Single-row chain: original images, each pair's prediction at its own (x, y)."""
    n = len(panels)
    if n == 0:
        return
    fig = plt.figure(figsize=(max(16, 4.2 * n), 11))
    ax = fig.add_axes([0.02, 0.22, 0.96, 0.64])
    _render_chain_row_on_ax(
        ax,
        panels,
        align_mode=align_mode,
        align_keypoint=align_keypoint,
        tps_line_control=tps_line_control,
        roi_size_ref_box=roi_size_ref_box,
    )
    if experiment_type:
        fig.text(0.5, 0.97, experiment_type, ha="center", va="top", fontsize=15, fontweight="bold")
    if experiment_detail:
        fig.text(0.5, 0.935, experiment_detail, ha="center", va="top", fontsize=10)
    foot = (
        "Original mammograms (not shifted). Orange dots = each pair's predicted point; "
        "orange path connects those points and can bend."
    )
    fig.text(0.5, 0.12, foot, ha="center", va="top", fontsize=9, color="0.25")
    if step_lines:
        fig.text(0.5, 0.02, "\n".join(step_lines[:10]), ha="center", va="bottom", fontsize=8)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_chain_overview_dual_row_figure(
    sequential_panels: list[ChainPanelInfo],
    anchor_panels: list[ChainPanelInfo],
    save_path: Path,
    *,
    experiment_type: str,
    experiment_detail: str,
    step_lines: Optional[list[str]] = None,
    align_keypoint: tuple[float, float] = (256.0, 180.0),
    align_mode: str = "raw",
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
) -> None:
    """Row 1: sequential exp5 chain. Row 2: ROI exam → each prior (same column dates as row 1)."""
    n = len(sequential_panels)
    if n == 0 or len(anchor_panels) != n:
        return
    fig = plt.figure(figsize=(max(16, 4.2 * n), 18))
    ax_top = fig.add_axes([0.02, 0.52, 0.96, 0.38])
    ax_bot = fig.add_axes([0.02, 0.14, 0.96, 0.38])
    row_kw = dict(
        align_mode=align_mode,
        align_keypoint=align_keypoint,
        roi_size_ref_box=roi_size_ref_box,
    )
    _render_chain_row_on_ax(
        ax_top,
        sequential_panels,
        row_title="Row 1 — Sequential chain (ROI exam → older; each hop from previous exam)",
        **row_kw,
    )
    _render_chain_row_on_ax(
        ax_bot,
        anchor_panels,
        row_title="Row 2 — ROI exam → each prior (direct; same older dates as row 1)",
        **row_kw,
    )
    if experiment_type:
        fig.text(0.5, 0.98, experiment_type, ha="center", va="top", fontsize=15, fontweight="bold")
    if experiment_detail:
        fig.text(0.5, 0.955, experiment_detail, ha="center", va="top", fontsize=10)
    foot = (
        "Original mammograms (not shifted). Orange dots = each pair's predicted point; "
        "orange path connects those points and can bend."
    )
    fig.text(0.5, 0.08, foot, ha="center", va="top", fontsize=9, color="0.25")
    if step_lines:
        fig.text(0.5, 0.02, "\n".join(step_lines[:8]), ha="center", va="bottom", fontsize=8)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    return text[:max_len].strip("_") or "run"


def pick_point_on_source(src_tensor: torch.Tensor, source_path: str) -> tuple[float, float]:
    """Display source (512) and return one click in pixel coords (x, y) for model input."""
    img = src_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(f"Source — click correspondence point\n{Path(source_path).name}")
    ax.set_xlabel("Click once, then close the window (or press Enter in terminal).")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise SystemExit("No point selected. Run again and click on the source image.")
    x, y = pts[0]
    x = float(np.clip(x, 0, 511))
    y = float(np.clip(y, 0, 511))
    print(f"Selected source point (512 space): x={x:.1f}, y={y:.1f}")
    return x, y


class SinglePairDataset(Dataset):
    def __init__(self, src: torch.Tensor, trg: torch.Tensor, src_xy: tuple[float, float]):
        self.src = src
        self.trg = trg
        sx, sy = src_xy
        self.src_kps = torch.tensor([[sx], [sy]], dtype=torch.float32)
        # No ground truth on target — placeholder for dataloader shape
        self.trg_kps = torch.tensor([[-1.0], [-1.0]], dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "pckthres": torch.tensor([512.0]),
            "src_img": self.src,
            "trg_img": self.trg,
            "src_kps": self.src_kps,
            "trg_kps": self.trg_kps,
            "n_pts": torch.tensor([1]),
            "idx": torch.tensor([0]),
        }


def _file_dialog_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    root.update()
    return root


def browse_image_file(title: str) -> str:
    from tkinter import filedialog

    root = _file_dialog_root()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit(f"Cancelled — no source/target file selected ({title}).")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Not a file: {p}")
    return str(p.resolve())


def browse_output_dir(title: str, initial_dir: Path | None = None) -> str:
    from tkinter import filedialog

    start = str(initial_dir.resolve()) if initial_dir and initial_dir.exists() else str(ROOT / "outputs")
    root = _file_dialog_root()
    path = filedialog.askdirectory(title=title, initialdir=start, mustexist=False)
    root.destroy()
    if not path:
        fallback = initial_dir or (ROOT / "outputs" / "mammo_runs")
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"No folder chosen — using default: {fallback}")
        return str(fallback.resolve())
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def prompt_path_terminal(label: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label}{hint}: ").strip()
        if not raw and default:
            raw = default
        if not raw:
            print("  Path required.")
            continue
        p = Path(raw).expanduser()
        if label.lower().startswith("output") or "save" in label.lower():
            p.mkdir(parents=True, exist_ok=True)
            return str(p.resolve())
        if not p.is_file():
            print(f"  File not found: {p}")
            continue
        return str(p.resolve())


def write_run_metadata(
    out_dir: Path,
    prefix: str,
    title: str,
    description: str,
    source_path: str,
    target_path: str,
    src_xy: tuple[float, float],
):
    meta = {
        "title": title,
        "description": description,
        "source_image": source_path,
        "target_image": target_path,
        "source_point_512": {"x": src_xy[0], "y": src_xy[1]},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
    }
    with open(out_dir / f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / f"{prefix}_README.txt", "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n\n")
        f.write(f"Description:\n{description}\n\n")
        f.write(f"Source: {source_path}\n")
        f.write(f"Target: {target_path}\n")
        f.write(f"Source point (512 px): x={src_xy[0]:.2f}, y={src_xy[1]:.2f}\n")


def run_correspondence(
    ldm,
    mini_batch,
    *,
    save_folder: Path,
    file_stem: str,
    source_path: str,
    target_path: str,
    src_display: torch.Tensor,
    trg_display: torch.Tensor,
    trg_gt_box: Optional[tuple[float, float, float, float]],
    src_gt_box: Optional[tuple[float, float, float, float]] = None,
    src_all_gt_boxes: Optional[list] = None,
    trg_all_gt_boxes: Optional[list] = None,
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
    pairs_save_path: Optional[Path] = None,
    device: str,
    upsample_res: int,
    num_steps: int,
    noise_level: int,
    layers: list[int],
    lr: float,
    num_opt_iterations: int,
    num_iterations: int,
    sigma: float,
    flip_prob: float,
    crop_percent: float,
    experiment_type: str = "",
    experiment_detail: str = "",
    gt_mode: str = "gaussian",
):
    """One source point; target GT box optional (512 xyxy)."""
    j = 0
    src_kp = mini_batch["src_kps"][0, :, j]
    contexts = []
    for _ in range(num_opt_iterations):
        context = optimize_prompt(
            ldm,
            mini_batch["src_img"][0],
            src_kp / 512,
            num_steps=num_steps,
            device=device,
            layers=layers,
            lr=lr,
            upsample_res=upsample_res,
            noise_level=noise_level,
            sigma=sigma,
            flip_prob=flip_prob,
            crop_percent=crop_percent,
            gt_mode=gt_mode,
            gt_box_xyxy=src_gt_box,
        )
        contexts.append(context)
    if src_gt_box is not None and str(gt_mode).lower().replace("-", "_") in {
        "roi",
        "roi_box",
        "box",
        "binary",
    }:
        box_norm = (
            src_gt_box[0] / 512.0,
            src_gt_box[1] / 512.0,
            src_gt_box[2] / 512.0,
            src_gt_box[3] / 512.0,
        )
        viz_dev = src_kp.device if hasattr(src_kp, "device") else device
        roi_gt = roi_box_mask(box_norm, size=upsample_res, device=viz_dev)
        gauss_gt = gaussian_circle(
            src_kp / 512, size=upsample_res, sigma=sigma, device=viz_dev
        )
        kp_viz = src_kp * (upsample_res / 512.0)
        visualize_image_with_points(
            roi_gt[None], kp_viz, f"{file_stem}_gt_roi_box_mask", save_folder=str(save_folder)
        )
        visualize_image_with_points(
            gauss_gt[None], kp_viz, f"{file_stem}_gt_gaussian_ref", save_folder=str(save_folder)
        )

    all_maps = []
    for context in contexts:
        maps = []
        attn_maps, _ = run_image_with_tokens_cropped(
            ldm,
            mini_batch["trg_img"][0],
            context,
            index=0,
            upsample_res=upsample_res,
            noise_level=noise_level,
            layers=layers,
            device=device,
            crop_percent=crop_percent,
            num_iterations=num_iterations,
        )
        for k in range(attn_maps.shape[0]):
            avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
            maps.append(avg)
        all_maps.append(torch.stack(maps, dim=0))
    all_maps = torch.stack(all_maps, dim=0)
    all_maps = torch.mean(all_maps, dim=0)
    all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res * upsample_res))
    all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)

    stem = file_stem
    visualize_image_with_points(
        mini_batch["src_img"][0], src_kp, f"{stem}_initial_point_{j:02d}", save_folder=str(save_folder)
    )

    for k in range(all_maps.shape[0]):
        visualize_image_with_points(
            all_maps[k, None], None, f"{stem}_largest_loc_trg_{j:02d}_{k:02d}", save_folder=str(save_folder)
        )
    visualize_image_with_points(
        torch.mean(all_maps, dim=0)[None], None, f"{stem}_largest_loc_trg_{j:02d}_mean", save_folder=str(save_folder)
    )

    mean_trg_map = torch.mean(all_maps, dim=0)
    est = find_max_pixel_value(mean_trg_map, img_size=512) + 0.5
    est_keypoints = torch.zeros_like(mini_batch["src_kps"])
    est_keypoints[0, :, j] = est

    trg_pred_box = None
    trg_heatmap_box = None
    trg_peak_sum_box = None
    trg_pred_white_box = None
    roi_iou_pred = None
    roi_iou_heatmap = None
    roi_iou_peak_sum = None
    hm_cx, hm_cy = 256.0, 256.0
    pk_cx, pk_cy = 256.0, 256.0
    center_error = None
    native_wh = _png_wh(target_path)

    if trg_gt_box is not None:
        gx1, gy1, gx2, gy2 = trg_gt_box
        gw = gx2 - gx1
        gh = gy2 - gy1
        hm_cx, hm_cy = heatmap_max_sum_box_center(mean_trg_map, gw, gh)
        pk_cx, pk_cy = heatmap_max_sum_box_center_with_peak(mean_trg_map, gw, gh)
        trg_pred_box = box_from_center_size(est[0].item(), est[1].item(), gw, gh)
        trg_heatmap_box = box_from_center_size(hm_cx, hm_cy, gw, gh)
        trg_peak_sum_box = box_from_center_size(pk_cx, pk_cy, gw, gh)
        roi_iou_pred = box_iou_xyxy(trg_gt_box, trg_pred_box)
        roi_iou_heatmap = box_iou_xyxy(trg_gt_box, trg_heatmap_box)
        roi_iou_peak_sum = box_iou_xyxy(trg_gt_box, trg_peak_sum_box)
        print(
            f"GT ROI center: ({(gx1+gx2)/2:.1f}, {(gy1+gy2)/2:.1f})  size: {gw:.1f} x {gh:.1f} px"
        )
        print(f"Pred point center: ({est[0].item():.1f}, {est[1].item():.1f})")
        print(f"Heatmap max-sum ROI center: ({hm_cx:.1f}, {hm_cy:.1f})")
        print(f"Max-sum+peak ROI center: ({pk_cx:.1f}, {pk_cy:.1f})")
        print(f"ROI IoU vs GT — pred box (blue): {roi_iou_pred:.4f}")
        pred_xy = (est[0].item(), est[1].item())
        center_error = {
            "vs_primary_gt": center_error_metrics(
                trg_gt_box, pred_xy, native_wh=native_wh
            ),
            "vs_heatmap_max_sum": center_error_metrics(
                trg_gt_box, (hm_cx, hm_cy), native_wh=native_wh
            ),
            "vs_heatmap_peak_sum": center_error_metrics(
                trg_gt_box, (pk_cx, pk_cy), native_wh=native_wh
            ),
        }
        pool = list(trg_all_gt_boxes or [])
        if not any(_boxes_close(trg_gt_box, b) for b in pool):
            pool.insert(0, trg_gt_box)
        nearest = nearest_center_error(pool, pred_xy, native_wh=native_wh)
        if nearest is not None:
            center_error["vs_nearest_gt"] = nearest
        pe = center_error["vs_primary_gt"]
        print(format_center_error_line(pe))
        if "dist_native_px" in pe:
            print(
                f"  dx,dy 512=({pe['dx_512']:.1f},{pe['dy_512']:.1f})  "
                f"native=({pe['dx_native_px']:.1f},{pe['dy_native_px']:.1f})  "
                f"{pe['dist_pct_of_width']:.2f}% of 512-W  "
                f"{pe['dist_pct_of_native_width']:.2f}% of PNG-W"
            )
    else:
        size_ref = src_gt_box if src_gt_box is not None else roi_size_ref_box
        if size_ref is not None:
            sx1, sy1, sx2, sy2 = size_ref
            gw = sx2 - sx1
            gh = sy2 - sy1
            trg_pred_white_box = box_from_center_size(est[0].item(), est[1].item(), gw, gh)
            print(
                f"No target GT — white pred ROI (ref size {gw:.1f}×{gh:.1f}) "
                f"center: ({est[0].item():.1f}, {est[1].item():.1f})"
            )
        else:
            print("Warning: no target GT and no ROI size reference — IoU / white box skipped.")

    all_maps_src = []
    for context in contexts:
        attn_map_src, _ = run_image_with_tokens_cropped(
            ldm,
            mini_batch["src_img"][0],
            context,
            index=0,
            upsample_res=upsample_res,
            noise_level=noise_level,
            layers=layers,
            device=device,
            crop_percent=crop_percent,
            num_iterations=num_iterations,
        )
        maps = [torch.mean(attn_map_src[k], dim=0, keepdim=True) for k in range(attn_map_src.shape[0])]
        all_maps_src.append(torch.stack(maps, dim=0))
    all_maps_src = torch.mean(torch.stack(all_maps_src, dim=0), dim=0)
    all_maps_src = torch.nn.Softmax(dim=-1)(
        all_maps_src.reshape(len(layers), upsample_res * upsample_res)
    ).reshape(len(layers), upsample_res, upsample_res)
    for k in range(all_maps_src.shape[0]):
        visualize_image_with_points(
            all_maps_src[k, None],
            src_kp / 512 * upsample_res,
            f"{stem}_largest_loc_src_{j:02d}_{k:02d}",
            save_folder=str(save_folder),
        )
    visualize_image_with_points(
        torch.mean(all_maps_src, dim=0)[None], None, f"{stem}_largest_loc_src_{j:02d}_mean", save_folder=str(save_folder)
    )

    corr_path = save_folder / f"{stem}_correspondences_estimated.png"
    src_eid = ""
    m_ex = re.search(r"_ex(\d+)", stem)
    if m_ex:
        src_eid = m_ex.group(1)
    save_correspondence_figure(
        src_display,
        trg_display,
        src_kp,
        est,
        source_name=source_path,
        target_name=target_path,
        save_path=corr_path,
        src_gt_box=src_gt_box,
        trg_gt_box=trg_gt_box,
        src_all_gt_boxes=src_all_gt_boxes,
        trg_all_gt_boxes=trg_all_gt_boxes,
        trg_pred_box=trg_pred_box,
        trg_heatmap_box=None,
        trg_peak_sum_box=None,
        trg_pred_white_box=trg_pred_white_box,
        roi_iou_pred=roi_iou_pred,
        roi_iou_heatmap=None,
        roi_iou_peak_sum=None,
        experiment_type=experiment_type,
        experiment_detail=experiment_detail,
        src_exam_id=src_eid,
        trg_exam_id=src_eid if "same" in experiment_type.lower() and "exam" in experiment_type.lower() else "",
        center_error=center_error["vs_primary_gt"] if center_error else None,
    )
    if pairs_save_path is not None:
        pairs_save_path = Path(pairs_save_path)
        pairs_save_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(corr_path, pairs_save_path)
    save_heatmap_with_rois(
        mean_trg_map,
        save_folder / f"{stem}_target_attention_with_rois.png",
        trg_gt_box,
        trg_pred_box,
        trg_heatmap_box,
        trg_peak_sum_box=trg_peak_sum_box,
        trg_pred_white_box=trg_pred_white_box,
        trg_display=trg_display,
    )

    heatmap_at_pred = sample_heatmap_at_xy(mean_trg_map, est[0].item(), est[1].item())
    heatmap_max = float(mean_trg_map.max().item())

    torch.save(
        {
            "est_keypoints": est_keypoints,
            "src_kps": mini_batch["src_kps"],
            "contexts": torch.stack(contexts),
            "trg_gt_roi_xyxy": trg_gt_box,
            "trg_pred_roi_xyxy": trg_pred_box,
            "trg_pred_white_roi_xyxy": trg_pred_white_box,
            "trg_heatmap_max_sum_roi_xyxy": trg_heatmap_box,
            "trg_peak_sum_roi_xyxy": trg_peak_sum_box,
            "heatmap_max_sum_center_512": {"x": hm_cx, "y": hm_cy},
            "heatmap_peak_sum_center_512": {"x": pk_cx, "y": pk_cy},
            "roi_iou_pred_point": roi_iou_pred,
            "roi_iou_heatmap_max_sum": roi_iou_heatmap,
            "roi_iou_heatmap_peak_sum": roi_iou_peak_sum,
            "heatmap_score_at_pred": heatmap_at_pred,
            "heatmap_map_max": heatmap_max,
            "center_error": center_error,
        },
        save_folder / f"{stem}_correspondence_data.pt",
    )
    (save_folder / f"{stem}_center_error.json").write_text(
        json.dumps(center_error, indent=2),
        encoding="utf-8",
    )
    print(
        f"Estimated target point (512 px): x={est[0].item():.2f}, y={est[1].item():.2f}  "
        f"| attn@point={heatmap_at_pred:.6f} (map max={heatmap_max:.6f})"
    )
    return (
        est,
        roi_iou_pred,
        roi_iou_heatmap,
        trg_gt_box,
        trg_pred_box,
        trg_heatmap_box,
        trg_peak_sum_box,
        roi_iou_peak_sum,
        trg_pred_white_box,
        heatmap_at_pred,
        heatmap_max,
    )


def should_run_roundtrip_back(
    *,
    src_gt_box: Optional[tuple[float, float, float, float]],
    trg_gt_box: Optional[tuple[float, float, float, float]],
) -> bool:
    """Always run step 2 (forward then back) for exp1–5."""
    return True


def run_roundtrip_back_to_source(
    ldm,
    *,
    trg_t: torch.Tensor,
    src_t: torch.Tensor,
    forward_target_kp: tuple[float, float],
    forward_src_kp: tuple[float, float],
    original_src_gt_box: tuple[float, float, float, float],
    original_src_path: str,
    original_trg_path: str,
    save_folder: Path,
    file_stem: str,
    device: str,
    experiment_type: str,
    experiment_detail: str,
    forward_trg_white_box: Optional[tuple[float, float, float, float]] = None,
    forward_trg_pred_box: Optional[tuple[float, float, float, float]] = None,
    src_all_gt_boxes: Optional[list] = None,
    trg_all_gt_boxes: Optional[list] = None,
    **hyper,
) -> dict:
    """
    Step 2: target (forward prediction as query) → original source; IoU vs source ROI.
    """
    save_folder.mkdir(parents=True, exist_ok=True)
    rt_stem = f"{file_stem}_roundtrip_back"
    forward_center_error = None
    forward_roi_iou = None
    fwd_ce_path = save_folder / f"{file_stem}_center_error.json"
    if fwd_ce_path.is_file():
        _fwd_ce = json.loads(fwd_ce_path.read_text(encoding="utf-8"))
        forward_center_error = (_fwd_ce or {}).get("vs_primary_gt")
    fwd_pt_path = save_folder / f"{file_stem}_correspondence_data.pt"
    if fwd_pt_path.is_file():
        _fwd = torch.load(fwd_pt_path, map_location="cpu", weights_only=False)
        forward_roi_iou = _fwd.get("roi_iou_pred_point")
    tx, ty = forward_target_kp
    rt_batch = next(
        iter(
            DataLoader(
                SinglePairDataset(trg_t, src_t, (float(tx), float(ty))),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    rt_detail = (
        f"{experiment_detail} | Step 2 round-trip: prior/target → original source "
        f"(IoU vs source ROI)"
    )
    (
        est_back,
        roi_iou_pred,
        roi_iou_heatmap,
        _gt,
        back_pred_box,
        back_heatmap_box,
        back_peak_box,
        roi_iou_peak,
        _white,
        hm_at,
        hm_max,
    ) = run_correspondence(
        ldm,
        rt_batch,
        save_folder=save_folder,
        file_stem=rt_stem,
        source_path=original_trg_path,
        target_path=original_src_path,
        src_display=trg_t,
        trg_display=src_t,
        src_gt_box=None,
        trg_gt_box=original_src_gt_box,
        src_all_gt_boxes=trg_all_gt_boxes,
        trg_all_gt_boxes=src_all_gt_boxes,
        roi_size_ref_box=original_src_gt_box,
        device=device,
        experiment_type=experiment_type,
        experiment_detail=rt_detail,
        **hyper,
    )
    bx, by = est_back[0].item(), est_back[1].item()
    back_center_error = None
    rt_ce_path = save_folder / f"{rt_stem}_center_error.json"
    if rt_ce_path.is_file():
        _rt_ce = json.loads(rt_ce_path.read_text(encoding="utf-8"))
        back_center_error = (_rt_ce or {}).get("vs_primary_gt")
    trg_gt_for_bidir = trg_all_gt_boxes[0] if trg_all_gt_boxes else None
    overlay_path = save_folder / f"{rt_stem}_source_roundtrip_overlay.png"
    save_roundtrip_source_overlay_figure(
        src_t,
        src_gt_box=original_src_gt_box,
        back_pred_box=back_pred_box,
        back_heatmap_box=back_heatmap_box,
        back_peak_sum_box=back_peak_box,
        back_kp=(bx, by),
        roi_iou_pred=roi_iou_pred,
        roi_iou_heatmap=roi_iou_heatmap,
        roi_iou_peak_sum=roi_iou_peak,
        source_name=str(original_src_path),
        save_path=overlay_path,
        experiment_type=experiment_type,
        experiment_detail=rt_detail,
        center_error=back_center_error,
    )
    bidir_path = save_folder / f"{file_stem}_bidirectional_pair.png"
    save_bidirectional_pair_figure(
        src_t,
        trg_t,
        forward_src_kp=forward_src_kp,
        forward_trg_kp=(float(tx), float(ty)),
        back_src_kp=(bx, by),
        source_name=str(original_src_path),
        target_name=str(original_trg_path),
        save_path=bidir_path,
        src_gt_box=original_src_gt_box,
        src_all_gt_boxes=src_all_gt_boxes,
        trg_all_gt_boxes=trg_all_gt_boxes,
        back_pred_box=back_pred_box,
        back_heatmap_box=back_heatmap_box,
        back_peak_sum_box=back_peak_box,
        forward_trg_pred_box=forward_trg_pred_box,
        trg_pred_white_box=forward_trg_white_box,
        trg_gt_box=trg_gt_for_bidir,
        forward_roi_iou=forward_roi_iou,
        back_roi_iou=roi_iou_pred,
        forward_center_error=forward_center_error,
        back_center_error=back_center_error,
        roi_iou_pred=roi_iou_pred,
        roi_iou_heatmap=roi_iou_heatmap,
        roi_iou_peak_sum=roi_iou_peak,
        experiment_type=experiment_type,
        experiment_detail=experiment_detail,
    )
    iou_path = save_folder / f"{file_stem}_roundtrip_iou.json"
    iou_path.write_text(
        json.dumps(
            {
                "roi_iou_pred_point": roi_iou_pred,
                "roi_iou_heatmap_max_sum": roi_iou_heatmap,
                "roi_iou_heatmap_peak_sum": roi_iou_peak,
                "back_source_kp_512": {"x": bx, "y": by},
                "forward_target_kp_512": {"x": float(tx), "y": float(ty)},
                "center_error": json.loads(
                    (save_folder / f"{rt_stem}_center_error.json").read_text(encoding="utf-8")
                )
                if (save_folder / f"{rt_stem}_center_error.json").is_file()
                else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    main_pair_path = save_folder / f"{file_stem}_correspondences_estimated.png"
    return {
        "forward_target_kp_512": {"x": float(tx), "y": float(ty)},
        "back_source_kp_512": {"x": bx, "y": by},
        "roi_iou_pred_point": roi_iou_pred,
        "roi_iou_heatmap_max_sum": roi_iou_heatmap,
        "roi_iou_heatmap_peak_sum": roi_iou_peak,
        "back_pred_roi_xyxy": list(back_pred_box) if back_pred_box else None,
        "heatmap_score_at_back_pred": hm_at,
        "heatmap_map_max": hm_max,
        "center_error": json.loads(
            (save_folder / f"{rt_stem}_center_error.json").read_text(encoding="utf-8")
        )
        if (save_folder / f"{rt_stem}_center_error.json").is_file()
        else None,
        "overlay_png": str(overlay_path),
        "bidirectional_pair_png": str(bidir_path),
        "main_pair_png": str(main_pair_path),
        "iou_json": str(iou_path),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Interactive correspondence on a source/target image pair.")
    p.add_argument("--source", type=str, help="Path to source image (optional; file dialog if omitted)")
    p.add_argument("--target", type=str, help="Path to target image (optional; file dialog if omitted)")
    p.add_argument("--out", type=str, help="Output directory (optional; folder dialog if omitted)")
    p.add_argument(
        "--terminal-paths",
        action="store_true",
        help="Type paths in the terminal instead of browse dialogs",
    )
    p.add_argument("--title", type=str, help="Short title for this run (optional; prompted if omitted)")
    p.add_argument("--description", type=str, default="", help="Longer description (optional; prompted if omitted)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_steps", type=int, default=129)
    p.add_argument("--noise_level", type=int, default=-8)
    p.add_argument("--num_opt_iterations", type=int, default=5)
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=0.0023755632081200314)
    p.add_argument("--sigma", type=float, default=27.97853316316864)
    p.add_argument(
        "--gt-mode",
        type=str,
        default="gaussian",
        choices=("gaussian", "roi_box"),
        help="Optimization GT: gaussian around the click/ROI center, or binary source ROI box.",
    )
    p.add_argument("--crop_percent", type=float, default=93.16549294381423)
    p.add_argument("--flip_prob", type=float, default=0.0)
    p.add_argument("--layers", type=int, nargs="+", default=[7, 8, 9, 10])
    p.add_argument("--model_type", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--upsample_res", type=int, default=512)
    p.add_argument(
        "--src-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Source point in 512-px coordinates (skips the click window)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("\n=== Interactive semantic correspondence ===\n")
    if args.source:
        source_path = str(Path(args.source).expanduser().resolve())
    elif args.terminal_paths:
        source_path = prompt_path_terminal("Path to SOURCE image")
    else:
        print("Select SOURCE image in the file dialog...")
        source_path = browse_image_file("Select SOURCE mammogram / image")

    if args.target:
        target_path = str(Path(args.target).expanduser().resolve())
    elif args.terminal_paths:
        target_path = prompt_path_terminal("Path to TARGET image")
    else:
        print("Select TARGET image in the file dialog...")
        target_path = browse_image_file("Select TARGET mammogram / image")

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    src_pack = load_image_pack(source_path)
    trg_pack = load_image_pack(target_path)
    print(f"Red ROI on source (for display): {'yes ' + str(src_pack.roi_box) if src_pack.roi_box else 'NOT detected'}")
    print(f"Red ROI on target (for IoU):     {'yes ' + str(trg_pack.roi_box) if trg_pack.roi_box else 'NOT detected'}")
    if trg_pack.roi_box is None:
        print(
            "  → Without target GT, IoU is skipped; pred box is blue. "
            "Ensure the TARGET PNG has a red box overlay."
        )

    if args.src_xy is not None:
        src_xy = (
            float(np.clip(args.src_xy[0], 0, 511)),
            float(np.clip(args.src_xy[1], 0, 511)),
        )
        print(f"Source point (512 space): x={src_xy[0]:.1f}, y={src_xy[1]:.1f}")
    else:
        src_xy = pick_point_on_source(src_pack.display, source_path)

    title = args.title or input("Title for this run (e.g. 'CC view pair patient A'): ").strip()
    if not title:
        title = Path(source_path).stem + "_to_" + Path(target_path).stem

    description = args.description
    if not description and not args.title:
        description = input("Description (optional): ").strip()

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    elif args.terminal_paths:
        out_dir = Path(prompt_path_terminal("Output directory to save results", default="outputs/mammo_runs"))
    else:
        print("Select folder to save results...")
        out_dir = Path(browse_output_dir("Select output folder", ROOT / "outputs" / "mammo_runs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{slugify(title)}_{ts}"

    dataset = SinglePairDataset(src_pack.clean, trg_pack.clean, src_xy)
    mini_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Stable Diffusion on {device} (first run may download weights)...")
    ldm = load_ldm(device, args.model_type)

    print("Running optimization + correspondence (this can take several minutes)...\n")
    (
        est,
        roi_iou_pred,
        roi_iou_heatmap,
        trg_gt_box,
        trg_pred_box,
        trg_heatmap_box,
        _trg_peak_box,
        _roi_iou_peak,
        _trg_white,
        _hm_at,
        _hm_max,
    ) = run_correspondence(
        ldm,
        mini_batch,
        save_folder=out_dir,
        file_stem=prefix,
        source_path=source_path,
        target_path=target_path,
        src_display=src_pack.display,
        trg_display=trg_pack.display,
        trg_gt_box=trg_pack.roi_box,
        device=device,
        upsample_res=args.upsample_res,
        num_steps=args.num_steps,
        noise_level=args.noise_level,
        layers=args.layers,
        lr=args.learning_rate,
        num_opt_iterations=args.num_opt_iterations,
        num_iterations=args.num_iterations,
        sigma=args.sigma,
        flip_prob=args.flip_prob,
        crop_percent=args.crop_percent,
        src_gt_box=src_pack.roi_box,
        gt_mode=args.gt_mode,
    )

    roundtrip_meta = None
    print("\nStep 2: backward (target prediction → source)...\n")
    src_box_rt = src_pack.roi_box or box_from_center_size(src_xy[0], src_xy[1], 48, 48)
    if should_run_roundtrip_back(src_gt_box=src_box_rt, trg_gt_box=trg_pack.roi_box):
        roundtrip_meta = run_roundtrip_back_to_source(
            ldm,
            trg_t=trg_pack.clean,
            src_t=src_pack.clean,
            forward_target_kp=(est[0].item(), est[1].item()),
            forward_src_kp=src_xy,
            original_src_gt_box=src_box_rt,
            original_src_path=source_path,
            original_trg_path=target_path,
            save_folder=out_dir,
            file_stem=prefix,
            device=device,
            experiment_type=title,
            experiment_detail=description or title,
            forward_trg_white_box=_trg_white,
            forward_trg_pred_box=trg_pred_box,
            upsample_res=args.upsample_res,
            num_steps=args.num_steps,
            noise_level=args.noise_level,
            layers=args.layers,
            lr=args.learning_rate,
            num_opt_iterations=args.num_opt_iterations,
            num_iterations=args.num_iterations,
            sigma=args.sigma,
            flip_prob=args.flip_prob,
            crop_percent=args.crop_percent,
        )

    write_run_metadata(
        out_dir, prefix, title, description, source_path, target_path, src_xy,
    )
    meta_path = out_dir / f"{prefix}_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["estimated_target_point_512"] = {"x": est[0].item(), "y": est[1].item()}
    meta["trg_gt_roi_xyxy"] = list(trg_gt_box) if trg_gt_box else None
    meta["trg_pred_roi_xyxy"] = list(trg_pred_box) if trg_pred_box else None
    meta["trg_heatmap_max_sum_roi_xyxy"] = list(trg_heatmap_box) if trg_heatmap_box else None
    meta["roi_iou_pred_point"] = roi_iou_pred
    meta["roi_iou_heatmap_max_sum"] = roi_iou_heatmap
    meta["center_error"] = json.loads(
        (out_dir / f"{prefix}_center_error.json").read_text(encoding="utf-8")
    ) if (out_dir / f"{prefix}_center_error.json").is_file() else None
    meta["roundtrip_back"] = roundtrip_meta
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Results in:\n  {out_dir}\n")
    print(f"  {prefix}_metadata.json")
    print(f"  {prefix}_README.txt")
    print(f"  {prefix}_correspondences_estimated.png")
    print(f"  {prefix}_largest_loc_trg_00_mean.png  (target attention)\n")


if __name__ == "__main__":
    main()
