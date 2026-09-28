"""Map EMBED ROI_coords onto pack PNGs for SemCorre GT boxes (512 xyxy).

Official schema (parse with ``ast.literal_eval``):
  ``ROI_coords`` → list of ``[ymin, xmin, ymax, xmax]`` in **native image pixels**
  (row = y, col = x), same slicing as HITI patch extraction
  ``volume[frame, y_min:y_max, x_min:x_max]``.
  See https://docs.hitilab.com/docs/datasets/embed/rois

This repo's ``roi_overlays_cancer5/roi_coords.csv`` uses column ``roi_coord`` with
that same 4-tuple; ``roi_center`` is ``(cx, cy) = ((xmin+xmax)/2, (ymin+ymax)/2)``.
Draw each box only on the PNG in ``image_path`` (pack ``index.json`` join rule).

PNG exports may be horizontally flipped vs DICOM-native boxes. If ``reference_center``
(``roi_center``) is given, we choose raw vs flipped X by closest center; otherwise
we fall back to tissue-mass heuristics (StableKeypointsPlus-style).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

RES = 512


def flip_roi_x(box: list[float], width: int) -> list[float]:
    ymin, xmin, ymax, xmax = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [ymin, float(width) - xmax, ymax, float(width) - xmin]


def _box_center_xy(box: list[float]) -> tuple[float, float]:
    ymin, xmin, ymax, xmax = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)


def _patch_mean_in_box(
    gray: np.ndarray, box: list[float], png_w: int, png_h: int
) -> float:
    ymin, xmin, ymax, xmax = map(float, box[:4])
    y0 = max(0, min(png_h - 1, int(round(ymin))))
    y1 = max(0, min(png_h, int(round(ymax))))
    x0 = max(0, min(png_w - 1, int(round(xmin))))
    x1 = max(0, min(png_w, int(round(xmax))))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def _pick_raw_or_flipped_box(
    raw: list[float],
    flipped: list[float],
    png_w: int,
    png_h: int,
    gray,
    *,
    reference_center: tuple[float, float] | None,
    min_tissue_mean: float = 5.0,
) -> list[float]:
    """Choose DICOM-native vs horizontally flipped box for this PNG."""
    mean_raw = _patch_mean_in_box(gray, raw, png_w, png_h) if gray is not None else None
    mean_flip = _patch_mean_in_box(gray, flipped, png_w, png_h) if gray is not None else None
    raw_ok = mean_raw is not None and mean_raw >= min_tissue_mean
    flip_ok = mean_flip is not None and mean_flip >= min_tissue_mean

    # Prefer the placement that actually sits on tissue. If both do, still
    # take the clearly brighter one (copied boxes from another PNG/size
    # often land on a thin padding strip with mean just above the cutoff).
    if mean_raw is not None and mean_flip is not None:
        if flip_ok and not raw_ok:
            return flipped
        if raw_ok and not flip_ok:
            return raw
        if mean_flip > mean_raw * 1.15 and mean_flip >= min_tissue_mean:
            return flipped
        if mean_raw > mean_flip * 1.15 and mean_raw >= min_tissue_mean:
            return raw

    if reference_center is not None:
        rcx, rcy = reference_center
        cx_r, cy_r = _box_center_xy(raw)
        cx_f, cy_f = _box_center_xy(flipped)
        dr = (cx_r - rcx) ** 2 + (cy_r - rcy) ** 2
        df = (cx_f - rcx) ** 2 + (cy_f - rcy) ** 2
        by_ref = flipped if df < dr else raw
        by_ref_ok = flip_ok if by_ref is flipped else raw_ok
        other = raw if by_ref is flipped else flipped
        other_ok = raw_ok if by_ref is flipped else flip_ok
        if by_ref_ok:
            return by_ref
        if other_ok:
            return other
        return by_ref

    if mean_raw is not None and mean_flip is not None and mean_flip > mean_raw * 1.15:
        return flipped

    ymin, xmin, ymax, xmax = map(float, raw[:4])
    x0 = max(0, min(png_w - 1, int(round(xmin))))
    x1 = max(0, min(png_w, int(round(xmax))))
    y0 = max(0, min(png_h - 1, int(round(ymin))))
    y1 = max(0, min(png_h, int(round(ymax))))
    fx0 = max(0, min(png_w - 1, int(round(flipped[1]))))
    fx1 = max(0, min(png_w, int(round(flipped[3]))))

    use = list(raw)
    if gray is not None and y1 > y0 and x1 > x0 and fx1 > fx0:
        patch = gray[y0:y1, x0:x1]
        fpatch = gray[y0:y1, fx0:fx1]
        mean_raw = float(patch.mean()) if patch.size else 0.0
        mean_flip = float(fpatch.mean()) if fpatch.size else 0.0
        if mean_flip > mean_raw * 1.15:
            use = flipped
        else:
            cx = 0.5 * (xmin + xmax)
            mid = png_w // 2
            left_mass = float(gray[:, :mid].sum())
            right_mass = float(gray[:, mid:].sum())
            breast_on_left = left_mass >= right_mass
            if breast_on_left and cx > 0.55 * png_w and mean_raw < 15.0:
                use = flipped
            elif (not breast_on_left) and cx < 0.45 * png_w and mean_raw < 15.0:
                use = flipped
    else:
        cx = 0.5 * (xmin + xmax)
        mid = png_w // 2
        left_mass = float(gray[:, : mid].sum()) if gray is not None else 0.0
        right_mass = float(gray[:, mid:].sum()) if gray is not None else 0.0
        breast_on_left = left_mass >= right_mass
        if breast_on_left and cx > 0.55 * png_w:
            use = flipped
    return use


def adjust_rois_for_aligned_png(
    rois: list[list[float]],
    png_w: int,
    png_h: int,
    gray_or_rgb=None,
    *,
    reference_centers: list[tuple[float, float] | None] | None = None,
) -> list[list[float]]:
    if not rois or png_w <= 1:
        return list(rois or [])

    gray = None
    if gray_or_rgb is not None:
        arr = np.asarray(gray_or_rgb)
        if arr.ndim == 3:
            gray = arr.mean(axis=2)
        else:
            gray = arr.astype(np.float32)

    out: list[list[float]] = []
    for i, box in enumerate(rois):
        if not box or len(box) < 4:
            continue
        ymin, xmin, ymax, xmax = map(float, box[:4])
        raw = [ymin, xmin, ymax, xmax]
        flipped = flip_roi_x(raw, png_w)
        ref = None
        if reference_centers and i < len(reference_centers):
            ref = reference_centers[i]
        use = _pick_raw_or_flipped_box(
            raw,
            flipped,
            png_w,
            png_h,
            gray,
            reference_center=ref,
            min_tissue_mean=5.0,
        )
        out.append(use)
    return out


def box_mean_intensity(
    gray: np.ndarray, box_yx: tuple[float, float, float, float]
) -> float:
    ymin, xmin, ymax, xmax = box_yx
    h, w = gray.shape[:2]
    y0 = max(0, min(h, int(round(ymin))))
    y1 = max(0, min(h, int(round(ymax))))
    x0 = max(0, min(w, int(round(xmin))))
    x1 = max(0, min(w, int(round(xmax))))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def embed_box_on_png(
    png_path: Path | str,
    box_yx: tuple[float, float, float, float],
    *,
    min_tissue_mean: float = 5.0,
    reference_center: tuple[float, float] | None = None,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float],
    bool,
    tuple[float, float, float, float],
]:
    """Adjust EMBED box on this PNG; return native box, center (cx,cy), on-tissue flag."""
    path = Path(png_path)
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    refs = [reference_center] if reference_center is not None else None
    adjusted = adjust_rois_for_aligned_png([list(box_yx)], w, h, gray, reference_centers=refs)
    if not adjusted:
        raise ValueError(f"Empty ROI after adjust for {path}")
    ymin, xmin, ymax, xmax = adjusted[0]
    native = (ymin, xmin, ymax, xmax)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    on_tissue = box_mean_intensity(gray, native) >= min_tissue_mean
    return native, (cx, cy), on_tissue, box_yx


def _native_to_512(
    png_w: int,
    png_h: int,
    native: tuple[float, float, float, float],
    center: tuple[float, float],
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    ymin, xmin, ymax, xmax = native
    sx, sy = RES / png_w, RES / png_h
    box512 = (xmin * sx, ymin * sy, xmax * sx, ymax * sy)
    pt512 = (center[0] * sx, center[1] * sy)
    return box512, pt512


def roi_for_semcorre(
    png_path: Path | str,
    box_yx: tuple[float, float, float, float],
    *,
    require_on_tissue: bool = False,
    min_tissue_mean: float = 5.0,
    reference_center: tuple[float, float] | None = None,
) -> tuple[
    Optional[tuple[float, float, float, float]],
    tuple[float, float],
    bool,
]:
    """512-space GT box (or None), keypoint, and on-tissue flag."""
    path = Path(png_path)
    with Image.open(path) as im:
        w, h = im.size
    native, center, on_tissue, _ = embed_box_on_png(
        path,
        box_yx,
        min_tissue_mean=min_tissue_mean,
        reference_center=reference_center,
    )
    box512, pt512 = _native_to_512(w, h, native, center)
    if require_on_tissue and not on_tissue:
        return None, pt512, False
    return box512, pt512, on_tissue
