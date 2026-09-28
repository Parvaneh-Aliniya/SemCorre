"""Redraw SemCorre pair PNGs from saved pair.json + correspondence_data.pt (no LDM).

Fixes older runs where bidirectional figures hid the forward target pred ROI or
overwrote the main forward overlay.

Usage (PC, after download):
  python scripts/regenerate_pair_figures.py \\
    --run-root outputs/batch_experiments/vista_exp1_views \\
    --reviews-dir outputs/batch_experiments/Experiments/views

Vista (pack on scratch):
  python scripts/regenerate_pair_figures.py \\
    --run-root $SCRATCH/semcorre_batch_outputs/batch_experiments/vista_exp1_views \\
    --pack-dir $SCRATCH/sk_review/roi_overlays_exp1_views
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_mammo_correspondence as batch  # noqa: E402
from interactive_correspondence import (  # noqa: E402
    box_from_center_size,
    save_bidirectional_pair_figure,
    save_correspondence_figure,
)

RES = 512
BANNER_H = 80


def _load_pt(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _est_xy_from_pt(data: dict) -> tuple[float, float] | None:
    ek = data.get("est_keypoints")
    if ek is not None:
        t = ek if isinstance(ek, torch.Tensor) else torch.tensor(ek)
        return float(t[0, 0, 0].item()), float(t[0, 1, 0].item())
    box = data.get("trg_pred_roi_xyxy")
    if box:
        x1, y1, x2, y2 = box
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return None


def _box_from_pt(data: dict, key: str) -> tuple[float, float, float, float] | None:
    v = data.get(key)
    if not v:
        return None
    return tuple(float(x) for x in v)


def _infer_trg_gt(
    meta: dict,
    *,
    center_err: dict | None,
    src_gt: tuple[float, float, float, float] | None,
    trg_pred: tuple[float, float, float, float] | None,
    trg_white: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Old pair.json sometimes omits trg_gt_box_512; center_error still has GT center."""
    if meta.get("trg_gt_box_512"):
        return tuple(float(x) for x in meta["trg_gt_box_512"])
    if center_err:
        gc = center_err.get("gt_center_512") or {}
        if "x" in gc and "y" in gc:
            ref = src_gt or trg_pred or trg_white
            if ref is not None:
                w, h = ref[2] - ref[0], ref[3] - ref[1]
                return box_from_center_size(float(gc["x"]), float(gc["y"]), w, h)
    return None


def resolve_image_path(
    path_str: str,
    *,
    pack_root: Path | None,
    reviews_root: Path | None,
) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if pack_root is not None:
        try:
            return batch.resolve_clean_path(pack_root, path_str.replace("\\", "/"))
        except FileNotFoundError:
            pass
        tail = p.name
        m = re.search(r"patient_(\d+)", path_str.replace("\\", "/"))
        if m:
            pid = m.group(1)
            hits = list((pack_root / f"patient_{pid}").rglob(tail))
            if hits:
                if len(hits) == 1:
                    return hits[0]
                return _pick_pack_image_hit(hits, path_str)
    if reviews_root is not None:
        m = re.search(r"patient_(\d+)", path_str.replace("\\", "/"))
        if not m:
            raise FileNotFoundError(path_str)
        pid = m.group(1)
        folder = reviews_root / f"patient_{pid}"
        lat_view = p.stem.split("_")
        # R_CC.png or 2018-08-01_R_CC_...
        for cand in folder.glob("*.png"):
            name = cand.name.upper()
            if "_R_CC" in name or name.endswith("_R_CC.PNG") or "_L_CC" in name:
                if p.stem.upper().endswith("_R_CC") and "_R_CC" in name:
                    return cand
                if p.stem.upper().endswith("_L_CC") and "_L_CC" in name:
                    return cand
            if "_R_MLO" in name or "_L_MLO" in name:
                if "MLO" in p.stem.upper() and "MLO" in name:
                    return cand
        # fallback: match laterality + view tokens from path
        parts = path_str.replace("\\", "/").split("/")
        fname = parts[-1] if parts else p.name
        lv = fname.replace(".png", "").split("_")
        if len(lv) >= 2:
            lat, view = lv[-2].upper(), lv[-1].upper()
            for cand in folder.glob("*.png"):
                if f"_{lat}_{view}_" in cand.name.upper() or cand.name.upper().endswith(
                    f"_{lat}_{view}.PNG"
                ):
                    return cand
    raise FileNotFoundError(f"Cannot resolve image: {path_str}")


def _score_pack_image_hit(hit: Path, path_str: str) -> tuple:
    s = str(hit).replace("\\", "/").lower()
    bad = any(
        x in s
        for x in (
            "overlays_512",
            "correspondence",
            "chain_overview",
            "step16",
            "batch_experiments",
            "semcorre_batch",
        )
    )
    is_exam = bool(re.search(r"/\d{4}-\d{2}-\d{2}/[lr]_(cc|mlo)\.png", s))
    return (bad, not is_exam, len(s))


def _pick_pack_image_hit(hits: list[Path], path_str: str) -> Path:
    norm = path_str.replace("\\", "/")
    m = re.search(r"/(\d{4}-\d{2}-\d{2})/([LR])_([A-Z]+)\.png", norm, re.I)
    if m:
        want = f"{m.group(1)}/{m.group(2).upper()}_{m.group(3).upper()}.png".lower()
        for h in hits:
            if want in str(h).replace("\\", "/").lower():
                return h
    return min(hits, key=lambda h: _score_pack_image_hit(h, path_str))


def load_display_chw(path: Path) -> torch.Tensor:
    """512×512 display matching batch load_image_chw (full PNG stretch to 512)."""
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


def regen_one(pair_dir: Path, *, pack_root: Path | None, reviews_root: Path | None) -> bool:
    pair_jsons = list(pair_dir.glob("*_pair.json"))
    if not pair_jsons:
        return False
    meta = json.loads(pair_jsons[0].read_text(encoding="utf-8"))
    stem = meta["stem"]
    fwd_pt = _load_pt(pair_dir / f"{stem}_correspondence_data.pt")
    if fwd_pt is None:
        print(f"  skip {pair_dir.name}: no {stem}_correspondence_data.pt")
        return False

    src_path = resolve_image_path(meta["src_path"], pack_root=pack_root, reviews_root=reviews_root)
    trg_path = resolve_image_path(meta["trg_path"], pack_root=pack_root, reviews_root=reviews_root)
    src_t = load_display_chw(src_path)
    trg_t = load_display_chw(trg_path)

    sx, sy = meta["src_xy_512"]
    est = _est_xy_from_pt(fwd_pt)
    if est is None:
        rt = meta.get("roundtrip_back") or {}
        fk = rt.get("forward_target_kp_512") or {}
        if fk:
            est = (float(fk["x"]), float(fk["y"]))
    if est is None:
        print(f"  skip {pair_dir.name}: no forward target point")
        return False

    src_gt = tuple(meta["src_gt_box_512"]) if meta.get("src_gt_box_512") else None
    src_all = [tuple(b) for b in meta.get("src_all_gt_512") or []]
    trg_all = [tuple(b) for b in meta.get("trg_all_gt_512") or []]
    trg_pred = _box_from_pt(fwd_pt, "trg_pred_roi_xyxy")
    trg_white = _box_from_pt(fwd_pt, "trg_pred_white_roi_xyxy")
    center_err = None
    ce_path = pair_dir / f"{stem}_center_error.json"
    if ce_path.is_file():
        ce = json.loads(ce_path.read_text(encoding="utf-8"))
        center_err = ce.get("vs_primary_gt")
    trg_gt = _infer_trg_gt(
        meta,
        center_err=center_err,
        src_gt=src_gt,
        trg_pred=trg_pred,
        trg_white=trg_white,
    )
    if trg_gt is not None and not trg_all:
        trg_all = [trg_gt]
    forward_iou = fwd_pt.get("roi_iou_pred_point")
    if forward_iou is None:
        forward_iou = meta.get("roi_iou_pred_point")

    m_ex = re.search(r"_ex(\d+)", stem)
    exam_id = m_ex.group(1) if m_ex else ""

    save_correspondence_figure(
        src_t,
        trg_t,
        torch.tensor([sx, sy]),
        torch.tensor([est[0], est[1]]),
        source_name=str(src_path),
        target_name=str(trg_path),
        save_path=pair_dir / f"{stem}_correspondences_estimated.png",
        src_gt_box=src_gt,
        trg_gt_box=trg_gt,
        src_all_gt_boxes=src_all,
        trg_all_gt_boxes=trg_all,
        trg_pred_box=trg_pred,
        trg_heatmap_box=None,
        trg_peak_sum_box=None,
        trg_pred_white_box=trg_white,
        roi_iou_pred=forward_iou,
        experiment_type=meta.get("experiment_label", ""),
        experiment_detail=meta.get("experiment_detail", ""),
        src_exam_id=exam_id,
        trg_exam_id=exam_id,
        center_error=center_err,
    )

    rt = meta.get("roundtrip_back")
    if not rt:
        return True

    rt_pt = _load_pt(pair_dir / f"{stem}_roundtrip_back_correspondence_data.pt")
    back_pred = tuple(rt["back_pred_roi_xyxy"]) if rt.get("back_pred_roi_xyxy") else None
    if back_pred is None and rt_pt:
        back_pred = _box_from_pt(rt_pt, "trg_pred_roi_xyxy")

    fk = rt.get("forward_target_kp_512") or {}
    bk = rt.get("back_source_kp_512") or {}
    back_iou = rt.get("roi_iou_pred_point")
    back_ce = (rt.get("center_error") or {}).get("vs_primary_gt")
    rt_ce_path = pair_dir / f"{stem}_roundtrip_back_center_error.json"
    if rt_ce_path.is_file() and back_ce is None:
        back_ce = json.loads(rt_ce_path.read_text(encoding="utf-8")).get("vs_primary_gt")
    save_bidirectional_pair_figure(
        src_t,
        trg_t,
        forward_src_kp=(float(sx), float(sy)),
        forward_trg_kp=(float(fk.get("x", est[0])), float(fk.get("y", est[1]))),
        back_src_kp=(float(bk.get("x", sx)), float(bk.get("y", sy))),
        source_name=str(src_path),
        target_name=str(trg_path),
        save_path=pair_dir / f"{stem}_bidirectional_pair.png",
        src_gt_box=src_gt,
        src_all_gt_boxes=src_all,
        trg_all_gt_boxes=trg_all,
        trg_gt_box=trg_gt,
        back_pred_box=back_pred,
        forward_trg_pred_box=trg_pred,
        trg_pred_white_box=trg_white,
        forward_roi_iou=forward_iou,
        back_roi_iou=back_iou,
        forward_center_error=center_err,
        back_center_error=back_ce,
        experiment_type=meta.get("experiment_label", ""),
        experiment_detail=meta.get("experiment_detail", ""),
    )

    if rt_pt is not None and back_pred is not None:
        bx, by = float(bk.get("x", sx)), float(bk.get("y", sy))
        tx, ty = float(fk.get("x", est[0])), float(fk.get("y", est[1]))
        save_correspondence_figure(
            trg_t,
            src_t,
            torch.tensor([tx, ty]),
            torch.tensor([bx, by]),
            source_name=str(trg_path),
            target_name=str(src_path),
            save_path=pair_dir / f"{stem}_roundtrip_back_correspondences_estimated.png",
            src_gt_box=trg_gt,
            trg_gt_box=src_gt,
            src_all_gt_boxes=trg_all,
            trg_all_gt_boxes=src_all,
            trg_pred_box=back_pred,
            trg_heatmap_box=None,
            trg_peak_sum_box=None,
            trg_pred_white_box=None,
            experiment_type=meta.get("experiment_label", ""),
            experiment_detail=(meta.get("experiment_detail", "") + " | round-trip back"),
            src_exam_id=exam_id,
            trg_exam_id=exam_id,
            center_error=back_ce,
            roi_iou_pred=back_iou,
        )
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=str, required=True, help="e.g. vista_exp1_views")
    p.add_argument("--pack-dir", type=str, default="", help="Vista roi_overlays pack")
    p.add_argument(
        "--reviews-dir",
        type=str,
        default="",
        help="Experiments/views on PC (patient_*/date_Lat_view_*.png)",
    )
    p.add_argument("--patient", type=str, default="", help="Optional filter, e.g. 22911591")
    args = p.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    pack_root = Path(args.pack_dir).expanduser().resolve() if args.pack_dir else None
    reviews_root = Path(args.reviews_dir).expanduser().resolve() if args.reviews_dir else None
    if pack_root is None and reviews_root is None:
        raise SystemExit("Need --pack-dir and/or --reviews-dir")

    n_ok = 0
    for pair_dir in sorted(run_root.rglob("*")):
        if not pair_dir.is_dir():
            continue
        if args.patient and args.patient not in pair_dir.name:
            continue
        if not list(pair_dir.glob("*_pair.json")):
            continue
        try:
            if regen_one(pair_dir, pack_root=pack_root, reviews_root=reviews_root):
                n_ok += 1
                print("redrew", pair_dir.relative_to(run_root))
        except Exception as exc:
            print("FAILED", pair_dir, exc)
    print(f"Done: {n_ok} pair folder(s) under {run_root}")


if __name__ == "__main__":
    main()
