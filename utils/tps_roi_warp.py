"""Thin-plate spline warp (scikit-image) using the same SemCorre source/target pair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import ThinPlateSplineTransform, warp

RES = 512
IMG_CORNER_KEYS = ("img_tl", "img_tr", "img_br", "img_bl")
ROI_INTERIOR_KEYS = ("roi_tl", "roi_tr", "roi_br", "roi_bl", "roi_center")
ControlMode = Literal["roi", "breast"]
DEFAULT_TPS_MODES: tuple[ControlMode, ...] = ("roi", "breast")
NUM_BREAST_CONTROL_POINTS = 10


@dataclass(frozen=True)
class TpsWarpResult:
    control_mode: ControlMode
    warped_rgb: np.ndarray
    warped_src_kp: tuple[float, float]
    figure_path: Path
    meta_path: Path
    control_keys: tuple[str, ...]


def tensor_chw_to_rgb01(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def rgb01_to_tensor_chw(rgb: np.ndarray) -> torch.Tensor:
    arr = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return torch.tensor(arr.transpose(2, 0, 1).copy())


def image_corner_points() -> dict[str, tuple[float, float]]:
    m = float(RES - 1)
    return {
        "img_tl": (0.0, 0.0),
        "img_tr": (m, 0.0),
        "img_br": (m, m),
        "img_bl": (0.0, m),
    }


def roi_corner_center_points(box_xyxy: tuple[float, float, float, float]) -> dict[str, tuple[float, float]]:
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return {
        "roi_tl": (x1, y1),
        "roi_tr": (x2, y1),
        "roi_br": (x2, y2),
        "roi_bl": (x1, y2),
        "roi_center": (cx, cy),
    }


def default_box_around_point(xy: tuple[float, float], size: float = 48.0) -> tuple[float, float, float, float]:
    x, y = xy
    h = size * 0.5
    return (x - h, y - h, x + h, y + h)


def box_same_size_at(
    ref_box: Optional[tuple[float, float, float, float]],
    center_xy: tuple[float, float],
) -> tuple[float, float, float, float]:
    if ref_box is None:
        return default_box_around_point(center_xy)
    x1, y1, x2, y2 = ref_box
    w, h = x2 - x1, y2 - y1
    cx, cy = center_xy
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def breast_tissue_mask(rgb01: np.ndarray, *, min_gray: float = 0.06) -> np.ndarray:
    """Foreground breast on 512² mammogram (heuristic threshold + largest component)."""
    from scipy import ndimage

    gray = np.mean(rgb01, axis=2)
    mask = gray > min_gray
    mask = ndimage.binary_fill_holes(mask)
    labeled, n = ndimage.label(mask)
    if n < 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    mask = labeled == keep
    mask = ndimage.binary_erosion(mask, iterations=2)
    return mask


def uniform_points_in_mask(
    mask: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Spread n control points across breast tissue (farthest-point sampling)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        cx, cy = RES * 0.5, RES * 0.5
        return [(cx, cy)] * n
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rng = rng or np.random.default_rng(0)
    if len(coords) <= n:
        pts = [tuple(c) for c in coords]
        while len(pts) < n:
            pts.append(pts[-1])
        return pts[:n]

    cy_m = float(np.mean(ys))
    cx_m = float(np.mean(xs))
    dist0 = (coords[:, 0] - cx_m) ** 2 + (coords[:, 1] - cy_m) ** 2
    first = int(np.argmin(dist0))
    selected = [coords[first]]
    min_d2 = np.sum((coords - selected[0]) ** 2, axis=1)

    for _ in range(n - 1):
        nxt = int(np.argmax(min_d2))
        selected.append(coords[nxt])
        min_d2 = np.minimum(min_d2, np.sum((coords - coords[nxt]) ** 2, axis=1))

    return [(float(x), float(y)) for x, y in selected]


def breast_control_points(rgb01: np.ndarray, n: int = NUM_BREAST_CONTROL_POINTS) -> dict[str, tuple[float, float]]:
    mask = breast_tissue_mask(rgb01)
    pts = uniform_points_in_mask(mask, n)
    return {f"breast_{i:02d}": pts[i] for i in range(n)}


def control_keys_for_mode(mode: ControlMode) -> tuple[str, ...]:
    if mode == "roi":
        return IMG_CORNER_KEYS + ROI_INTERIOR_KEYS
    return IMG_CORNER_KEYS + tuple(f"breast_{i:02d}" for i in range(NUM_BREAST_CONTROL_POINTS))


def build_source_control_points(
    mode: ControlMode,
    *,
    rgb01: np.ndarray,
    roi_box: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
    pts = dict(image_corner_points())
    if mode == "roi":
        pts.update(roi_corner_center_points(roi_box))
    else:
        pts.update(breast_control_points(rgb01))
    return pts


def destination_points_from_semcorre(
    src_points: dict[str, tuple[float, float]],
    method_src: tuple[float, float],
    method_dst: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    dx = method_dst[0] - method_src[0]
    dy = method_dst[1] - method_src[1]
    dst = dict(image_corner_points())
    for k, (sx, sy) in src_points.items():
        if k in IMG_CORNER_KEYS:
            continue
        dst[k] = (sx + dx, sy + dy)
    return dst


def fit_tps_and_warp(
    src_rgb: np.ndarray,
    src_points: dict[str, tuple[float, float]],
    dst_points: dict[str, tuple[float, float]],
    control_keys: Sequence[str],
) -> tuple[np.ndarray, dict[str, tuple[float, float]], ThinPlateSplineTransform]:
    src_arr = np.array([src_points[k] for k in control_keys], dtype=np.float64)
    dst_arr = np.array([dst_points[k] for k in control_keys], dtype=np.float64)
    tps = ThinPlateSplineTransform()
    tps.estimate(src_arr, dst_arr)
    warped = warp(
        src_rgb,
        tps,
        output_shape=(RES, RES),
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    warped = np.clip(warped, 0.0, 1.0).astype(np.float32)
    warped_points: dict[str, tuple[float, float]] = {}
    for k in control_keys:
        out = tps(np.array([src_points[k]], dtype=np.float64))[0]
        warped_points[k] = (float(out[0]), float(out[1]))
    return warped, warped_points, tps


def warp_xy(tps: ThinPlateSplineTransform, xy: tuple[float, float]) -> tuple[float, float]:
    out = tps(np.array([[xy[0], xy[1]]], dtype=np.float64))[0]
    return float(out[0]), float(out[1])


def warp_box_xyxy_with_tps(
    tps: ThinPlateSplineTransform,
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    xs: list[float] = []
    ys: list[float] = []
    for c in corners:
        wx, wy = warp_xy(tps, c)
        xs.append(wx)
        ys.append(wy)
    return (min(xs), min(ys), max(xs), max(ys))


def tps_align_image_keypoint_to_ref(
    src_rgb: np.ndarray,
    method_src: tuple[float, float],
    ref_xy: tuple[float, float],
    *,
    control_mode: ControlMode = "roi",
    roi_box: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float], ThinPlateSplineTransform]:
    """Non-rigid TPS on one image so SemCorre keypoint moves to ref (chain horizontal line)."""
    src_points = build_source_control_points(control_mode, rgb01=src_rgb, roi_box=roi_box)
    dst_points = destination_points_from_semcorre(src_points, method_src, ref_xy)
    keys = control_keys_for_mode(control_mode)
    warped_rgb, _, tps = fit_tps_and_warp(src_rgb, src_points, dst_points, keys)
    warped_kp = warp_xy(tps, method_src)
    return warped_rgb, warped_kp, tps


def _mode_figure_suffix(mode: ControlMode) -> str:
    return "roi" if mode == "roi" else "breast10"


def save_tps_before_after_figure(
    *,
    src_rgb: np.ndarray,
    trg_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    src_points: dict[str, tuple[float, float]],
    dst_points: dict[str, tuple[float, float]],
    warped_points: dict[str, tuple[float, float]],
    method_src: tuple[float, float],
    method_dst: tuple[float, float],
    save_path: Path,
    title: str,
    subtitle: str,
    control_mode: ControlMode,
) -> None:
    roi_keys = list(ROI_INTERIOR_KEYS)
    breast_keys = [f"breast_{i:02d}" for i in range(NUM_BREAST_CONTROL_POINTS)]
    interior_keys = roi_keys if control_mode == "roi" else breast_keys
    interior_color = "red" if control_mode == "roi" else "limegreen"
    interior_marker = "s" if control_mode == "roi" else "o"

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [
        (axes[0], src_rgb, "Before (source)", src_points, method_src, "SemCorre source"),
        (axes[1], warped_rgb, "After (TPS-warped source)", warped_points, method_dst, "SemCorre target"),
        (axes[2], trg_rgb, "Target (reference)", dst_points, method_dst, "SemCorre target"),
    ]
    for ax, img, panel_title, pts, method_xy, method_lbl in panels:
        ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        for k in interior_keys:
            if k not in pts:
                continue
            x, y = pts[k]
            ax.scatter(
                [x],
                [y],
                c=interior_color,
                s=45 if control_mode == "breast" else 55,
                marker=interior_marker,
                edgecolors="k",
                zorder=5,
            )
        for k in IMG_CORNER_KEYS:
            x, y = pts[k]
            ax.scatter([x], [y], c="dodgerblue", s=40, marker="^", edgecolors="k", zorder=5)
        ax.scatter(
            [method_xy[0]],
            [method_xy[1]],
            c="orange",
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            label=method_lbl,
            zorder=6,
        )
        ax.set_xlim(0, RES)
        ax.set_ylim(RES, 0)
        ax.set_axis_off()

    mode_lbl = "ROI corners + center" if control_mode == "roi" else f"{NUM_BREAST_CONTROL_POINTS} breast tissue points"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=11)
    fig.text(
        0.5,
        0.02,
        f"TPS controls: image corners + {mode_lbl}. "
        "Interior points shifted by the SemCorre source→target translation; "
        "source/target images match the SemCorre pair.",
        ha="center",
        fontsize=9,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_tps_warp(
    *,
    src_display: torch.Tensor,
    trg_display: torch.Tensor,
    method_src: tuple[float, float],
    method_dst: tuple[float, float],
    save_folder: Path,
    file_stem: str,
    control_mode: ControlMode = "roi",
    src_gt_box: Optional[tuple[float, float, float, float]] = None,
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
    experiment_type: str = "",
    experiment_detail: str = "",
) -> TpsWarpResult:
    """Warp SemCorre **source** toward **target**; target image is reference only."""
    ref_box = roi_size_ref_box or src_gt_box
    box = src_gt_box if src_gt_box is not None else box_same_size_at(ref_box, method_src)
    src_rgb = tensor_chw_to_rgb01(src_display)
    trg_rgb = tensor_chw_to_rgb01(trg_display)
    src_points = build_source_control_points(control_mode, rgb01=src_rgb, roi_box=box)
    dst_points = destination_points_from_semcorre(src_points, method_src, method_dst)
    keys = control_keys_for_mode(control_mode)
    warped_rgb, warped_points, tps = fit_tps_and_warp(src_rgb, src_points, dst_points, keys)
    warped_src_kp = warp_xy(tps, method_src)

    suffix = _mode_figure_suffix(control_mode)
    png_path = save_folder / f"{file_stem}_tps_{suffix}_warp_before_after.png"
    save_tps_before_after_figure(
        src_rgb=src_rgb,
        trg_rgb=trg_rgb,
        warped_rgb=warped_rgb,
        src_points=src_points,
        dst_points=dst_points,
        warped_points=warped_points,
        method_src=method_src,
        method_dst=method_dst,
        save_path=png_path,
        title=experiment_type or f"TPS warp ({control_mode})",
        subtitle=experiment_detail,
        control_mode=control_mode,
    )

    meta_path = save_folder / f"{file_stem}_tps_{suffix}_warp_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "file_stem": file_stem,
                "control_mode": control_mode,
                "semcorre_source_xy512": {"x": method_src[0], "y": method_src[1]},
                "semcorre_target_xy512": {"x": method_dst[0], "y": method_dst[1]},
                "warped_source_kp_xy512": {"x": warped_src_kp[0], "y": warped_src_kp[1]},
                "figure": str(png_path),
                "num_breast_controls": NUM_BREAST_CONTROL_POINTS if control_mode == "breast" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TpsWarpResult(
        control_mode=control_mode,
        warped_rgb=warped_rgb,
        warped_src_kp=warped_src_kp,
        figure_path=png_path,
        meta_path=meta_path,
        control_keys=keys,
    )


def save_tps_warp_for_correspondence(
    *,
    src_display: torch.Tensor,
    trg_display: torch.Tensor,
    method_src: tuple[float, float],
    method_dst: tuple[float, float],
    save_folder: Path,
    file_stem: str,
    src_gt_box: Optional[tuple[float, float, float, float]] = None,
    roi_size_ref_box: Optional[tuple[float, float, float, float]] = None,
    experiment_type: str = "",
    experiment_detail: str = "",
    modes: Sequence[ControlMode] = ("roi",),
) -> dict[ControlMode, TpsWarpResult]:
    """Run one or more TPS variants (roi / breast) for the same SemCorre pair."""
    out: dict[ControlMode, TpsWarpResult] = {}
    for mode in modes:
        out[mode] = run_tps_warp(
            src_display=src_display,
            trg_display=trg_display,
            method_src=method_src,
            method_dst=method_dst,
            save_folder=save_folder,
            file_stem=file_stem,
            control_mode=mode,
            src_gt_box=src_gt_box,
            roi_size_ref_box=roi_size_ref_box,
            experiment_type=experiment_type,
            experiment_detail=experiment_detail,
        )
    return out

