"""
Thin-plate spline (TPS) warp baseline for mammogram experiment pairs (exp1–exp5).

Uses scikit-image ThinPlateSplineTransform (same API as the gallery example:
https://scikit-image.org/docs/stable/auto_examples/transform/plot_tps_deformation.html)

For each batch experiment pair:
  1. ROI in 512 space via EMBED helpers (utils/embed_roi through batch.roi_on_image) on step 1 / pair jobs.
  2. TPS control points: image corners + ROI corners + center (512×512).
  3. Target controls: one SemCorre keypoint (default) + translate ROI corners; optional 5× SemCorre with
     --semcorre-per-roi-point (very slow).
  4. skimage ThinPlateSplineTransform + warp → before/after PNG.

Speed (recommended after batch_mammo_correspondence exp5):
  python scripts/batch_mammo_tps_warp.py \\
    --reuse-semcorre-dir outputs/batch_experiments/YOUR_RUN_TAG \\
    --experiments exp5 --exp5-views 62877247:L:CC,45209155:R:MLO
  (no Stable Diffusion reload; reads target_kp_512 from *_chain.json)

Or fast LDM: --num_opt_iterations 3 --num_iterations 10 (defaults below).

Outputs:
  {out-dir}/{run-tag}/{experiment}/{pair-stem}/{pair-stem}_tps_roi_warp_before_after.png
  {pair-stem}_tps_roi_warp_meta.json

NOTE: TPS warp is built into batch_mammo_correspondence.py (--with-tps-warp, default on)
after each SemCorre pair. Use this script only to regenerate TPS from an old batch folder.

Run from repo root:
  python scripts/batch_mammo_tps_warp.py \\
    --pack-dir ../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5 \\
    --out-dir outputs/tps_warp \\
    --experiments exp1,exp2,exp3,exp4,exp5 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import ThinPlateSplineTransform, warp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import batch_mammo_correspondence as batch  # noqa: E402
from interactive_correspondence import (  # noqa: E402
    SinglePairDataset,
    load_ldm,
    slugify,
)
from utils.optimize_token import (  # noqa: E402
    find_max_pixel_value,
    optimize_prompt,
    run_image_with_tokens_cropped,
)

RES = 512


@dataclass
class WarpTask:
    experiment: str
    experiment_label: str
    detail: str
    stem: str
    src_path: Path
    trg_path: Path
    src_xy: tuple[float, float]
    src_gt_box: tuple[float, float, float, float] | None
    roi_size_ref_box: tuple[float, float, float, float] | None = None
    exp5_step: int = 0


def _tensor_chw_to_rgb01(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _roi_corner_center_points(
    box_xyxy: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
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


def _default_box_around_point(
    xy: tuple[float, float], size: float = 48.0
) -> tuple[float, float, float, float]:
    x, y = xy
    h = size * 0.5
    return (x - h, y - h, x + h, y + h)


def _box_same_size_at(
    ref_box: tuple[float, float, float, float] | None,
    center_xy: tuple[float, float],
) -> tuple[float, float, float, float]:
    if ref_box is None:
        return _default_box_around_point(center_xy)
    x1, y1, x2, y2 = ref_box
    w, h = x2 - x1, y2 - y1
    cx, cy = center_xy
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def try_reuse_semcorre_kp(reuse_root: Path | None, task: WarpTask) -> tuple[float, float] | None:
    """Load SemCorre target keypoint from a prior batch_mammo_correspondence run (exp5 chain.json)."""
    if reuse_root is None:
        return None
    root = reuse_root.expanduser().resolve()
    if not root.is_dir():
        return None

    if task.experiment == "exp5_sequential_prior" and task.exp5_step > 0:
        chain_stem = task.stem.split("_step", 1)[0]
        chain_json = root / "exp5_sequential_prior" / chain_stem / f"{chain_stem}_chain.json"
        if not chain_json.is_file():
            return None
        data = json.loads(chain_json.read_text(encoding="utf-8"))
        for step in data.get("steps") or []:
            if int(step.get("step", -1)) != task.exp5_step:
                continue
            kp = step.get("target_kp_512") or {}
            if "x" in kp and "y" in kp:
                return float(kp["x"]), float(kp["y"])
        return None

    return None


def _image_corner_points() -> dict[str, tuple[float, float]]:
    m = float(RES - 1)
    return {
        "img_tl": (0.0, 0.0),
        "img_tr": (m, 0.0),
        "img_br": (m, m),
        "img_bl": (0.0, m),
    }


def _stack_xy(points: dict[str, tuple[float, float]], keys: list[str]) -> np.ndarray:
    return np.array([points[k] for k in keys], dtype=np.float64)


def estimate_semcorre_target_xy(
    ldm,
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    src_xy: tuple[float, float],
    *,
    device: str,
    hyper: dict[str, Any],
) -> tuple[float, float]:
    """One source keypoint → argmax target location (512 px, x then y)."""
    mini_batch = next(
        iter(
            DataLoader(
                SinglePairDataset(src_t, trg_t, src_xy),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    src_kp = mini_batch["src_kps"][0, :, 0]
    contexts = []
    for _ in range(int(hyper["num_opt_iterations"])):
        context = optimize_prompt(
            ldm,
            mini_batch["src_img"][0],
            src_kp / RES,
            num_steps=int(hyper["num_steps"]),
            device=device,
            layers=hyper["layers"],
            lr=float(hyper["lr"]),
            upsample_res=int(hyper["upsample_res"]),
            noise_level=int(hyper["noise_level"]),
            sigma=float(hyper["sigma"]),
            flip_prob=float(hyper["flip_prob"]),
            crop_percent=float(hyper["crop_percent"]),
            gt_mode=hyper.get("gt_mode", "gaussian"),
            gt_box_xyxy=hyper.get("gt_box_xyxy"),
        )
        contexts.append(context)

    all_maps = []
    for context in contexts:
        maps = []
        attn_maps, _ = run_image_with_tokens_cropped(
            ldm,
            mini_batch["trg_img"][0],
            context,
            index=0,
            upsample_res=int(hyper["upsample_res"]),
            noise_level=int(hyper["noise_level"]),
            layers=hyper["layers"],
            device=device,
            crop_percent=float(hyper["crop_percent"]),
            num_iterations=int(hyper["num_iterations"]),
        )
        for k in range(attn_maps.shape[0]):
            maps.append(torch.mean(attn_maps[k], dim=0, keepdim=True))
        all_maps.append(torch.stack(maps, dim=0))
    all_maps = torch.mean(torch.stack(all_maps, dim=0), dim=0)
    all_maps = torch.nn.Softmax(dim=-1)(
        all_maps.reshape(len(hyper["layers"]), int(hyper["upsample_res"]) ** 2)
    ).reshape(len(hyper["layers"]), int(hyper["upsample_res"]), int(hyper["upsample_res"]))
    mean_map = torch.mean(all_maps, dim=0)
    est = find_max_pixel_value(mean_map, img_size=RES) + 0.5
    return float(est[0].item()), float(est[1].item())


def _destination_control_points(
    ldm,
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    src_points: dict[str, tuple[float, float]],
    method_src: tuple[float, float],
    *,
    device: str,
    hyper: dict[str, Any],
    per_roi_point_semcorre: bool,
    method_dst: tuple[float, float] | None = None,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """Target-space coordinates for TPS controls + SemCorre method point."""
    if method_dst is None:
        if ldm is None:
            raise ValueError("method_dst is required when LDM is not loaded")
        method_dst = estimate_semcorre_target_xy(
            ldm, src_t, trg_t, method_src, device=device, hyper=hyper
        )
    dx = method_dst[0] - method_src[0]
    dy = method_dst[1] - method_src[1]

    dst: dict[str, tuple[float, float]] = {}
    for k, v in _image_corner_points().items():
        dst[k] = v

    roi_keys = ["roi_tl", "roi_tr", "roi_br", "roi_bl", "roi_center"]
    if per_roi_point_semcorre:
        for k in roi_keys:
            dst[k] = estimate_semcorre_target_xy(
                ldm, src_t, trg_t, src_points[k], device=device, hyper=hyper
            )
    else:
        for k in roi_keys:
            sx, sy = src_points[k]
            dst[k] = (sx + dx, sy + dy)

    return dst, method_dst


def _fit_tps_and_warp(
    src_rgb: np.ndarray,
    src_xy_dict: dict[str, tuple[float, float]],
    dst_xy_dict: dict[str, tuple[float, float]],
    control_keys: list[str],
) -> tuple[np.ndarray, ThinPlateSplineTransform, np.ndarray]:
    src_arr = _stack_xy(src_xy_dict, control_keys)
    dst_arr = _stack_xy(dst_xy_dict, control_keys)
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
    return warped, tps, dst_arr


def _scatter_labeled(
    ax,
    points: dict[str, tuple[float, float]],
    keys: list[str],
    *,
    color: str,
    marker: str,
    size: float,
    label_prefix: str,
):
    first = True
    for k in keys:
        x, y = points[k]
        lbl = f"{label_prefix} {k}" if first else None
        ax.scatter([x], [y], c=color, s=size, marker=marker, linewidths=1.0, edgecolors="k", label=lbl, zorder=5)
        first = False


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
):
    roi_keys = ["roi_tl", "roi_tr", "roi_br", "roi_bl", "roi_center"]
    img_keys = ["img_tl", "img_tr", "img_br", "img_bl"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [
        (axes[0], src_rgb, "Before (source)", src_points, method_src, "SemCorre source"),
        (axes[1], warped_rgb, "After (TPS-warped source)", warped_points, method_dst, "SemCorre target"),
        (axes[2], trg_rgb, "Target (reference)", dst_points, method_dst, "SemCorre target"),
    ]
    for ax, img, panel_title, pts, method_xy, method_lbl in panels:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        _scatter_labeled(ax, pts, roi_keys, color="red", marker="s", size=55, label_prefix="ROI")
        _scatter_labeled(ax, pts, img_keys, color="dodgerblue", marker="^", size=40, label_prefix="Image")
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

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "TPS controls: 4 image corners + ROI corners + center. "
        "Orange = SemCorre keypoint. ROI target controls: translation from SemCorre center "
        "(use --semcorre-per-roi-point for independent ROI corner matches).",
        ha="center",
        fontsize=9,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pair_job_to_task(job: batch.PairJob) -> WarpTask:
    return WarpTask(
        experiment=job.experiment,
        experiment_label=job.experiment_label,
        detail=job.experiment_detail,
        stem=job.stem,
        src_path=job.src_path,
        trg_path=job.trg_path,
        src_xy=job.src_xy_512,
        src_gt_box=job.src_gt_box_512,
    )


def exp5_chain_anchor(
    chain: batch.ChainJob,
    pack_root: Path,
) -> tuple[Path, tuple[float, float], tuple[float, float, float, float] | None]:
    anchor_path = batch.resolve_clean_path(pack_root, chain.anchor_record.image_path)
    src_box, src_xy, _ = batch.roi_on_record(
        anchor_path, chain.anchor_record, require_on_tissue=False
    )
    return anchor_path, src_xy, src_box


def iter_exp5_trg_steps(
    chain: batch.ChainJob,
    pack_root: Path,
    index: list[dict],
) -> list[tuple[int, str, Path]]:
    """(step_index, trg_exam_date, trg_image_path) for each backward hop (view timeline)."""
    timeline = batch.view_exam_timeline(
        pack_root, index, chain.patient_id, chain.laterality, chain.view
    )
    if not timeline:
        timeline = list(chain.exam_dates)
    if chain.anchor_date not in timeline:
        return []
    anchor_idx = timeline.index(chain.anchor_date)
    if anchor_idx <= 0:
        return []
    out: list[tuple[int, str, Path]] = []
    for step, trg_date in enumerate(reversed(timeline[:anchor_idx]), start=1):
        trg_path = batch.resolve_exam_image(
            pack_root, chain.patient_id, trg_date, chain.laterality, chain.view
        )
        if trg_path is None:
            continue
        out.append((step, trg_date, trg_path))
    return out


def run_warp_task(
    ldm,
    task: WarpTask,
    run_root: Path,
    device: str,
    hyper: dict[str, Any],
    *,
    per_roi_point_semcorre: bool,
    reuse_root: Path | None = None,
    semcorre_mode: str = "ldm",
    fallback_ldm: bool = True,
) -> tuple[float, float] | None:
    """Run TPS + save figure. Returns SemCorre target point for chaining exp5."""
    out_dir = run_root / task.experiment / task.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    src_t = batch.load_image_chw(task.src_path)
    trg_t = batch.load_image_chw(task.trg_path)
    src_rgb = _tensor_chw_to_rgb01(src_t)
    trg_rgb = _tensor_chw_to_rgb01(trg_t)

    ref_box = task.roi_size_ref_box or task.src_gt_box
    box = task.src_gt_box if task.src_gt_box is not None else _box_same_size_at(ref_box, task.src_xy)
    src_points = {**_image_corner_points(), **_roi_corner_center_points(box)}
    method_src = task.src_xy

    method_dst_pre: tuple[float, float] | None = None
    semcorre_from = "ldm"
    if semcorre_mode in ("reuse", "auto") and reuse_root is not None:
        method_dst_pre = try_reuse_semcorre_kp(reuse_root, task)
        if method_dst_pre is not None:
            semcorre_from = "batch_reuse"
    if method_dst_pre is None and semcorre_mode == "reuse" and not fallback_ldm:
        print(f"  skip {task.stem}: no reused SemCorre kp in {reuse_root}", flush=True)
        return None
    if method_dst_pre is None and per_roi_point_semcorre and ldm is None:
        print(f"  skip {task.stem}: --semcorre-per-roi-point requires LDM", flush=True)
        return None

    dst_points, method_dst = _destination_control_points(
        ldm,
        src_t,
        trg_t,
        src_points,
        method_src,
        device=device,
        hyper=hyper,
        per_roi_point_semcorre=per_roi_point_semcorre,
        method_dst=method_dst_pre,
    )

    control_keys = ["img_tl", "img_tr", "img_br", "img_bl", "roi_tl", "roi_tr", "roi_br", "roi_bl", "roi_center"]
    warped_rgb, tps, _ = _fit_tps_and_warp(src_rgb, src_points, dst_points, control_keys)

    warped_points: dict[str, tuple[float, float]] = {}
    for k in control_keys:
        xy = np.array([src_points[k]], dtype=np.float64)
        out = tps(xy)[0]
        warped_points[k] = (float(out[0]), float(out[1]))

    png_path = out_dir / f"{task.stem}_tps_roi_warp_before_after.png"
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
        title=task.experiment_label,
        subtitle=task.detail,
    )

    meta = {
        "experiment": task.experiment,
        "stem": task.stem,
        "detail": task.detail,
        "src_path": str(task.src_path),
        "trg_path": str(task.trg_path),
        "method_src_xy512": {"x": method_src[0], "y": method_src[1]},
        "method_dst_xy512_semcorre": {"x": method_dst[0], "y": method_dst[1]},
        "semcorre_source": semcorre_from,
        "embed_roi_on_anchor": task.src_gt_box is not None,
        "roi_size_ref_from_embed": ref_box is not None,
        "control_keys": control_keys,
        "src_control_points": {k: {"x": v[0], "y": v[1]} for k, v in src_points.items()},
        "dst_control_points": {k: {"x": v[0], "y": v[1]} for k, v in dst_points.items()},
        "warped_roi_points": {k: {"x": v[0], "y": v[1]} for k, v in warped_points.items() if k.startswith("roi_")},
        "figure": str(png_path),
        "tps_reference": "https://scikit-image.org/docs/stable/auto_examples/transform/plot_tps_deformation.html",
    }
    (out_dir / f"{task.stem}_tps_roi_warp_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return method_dst


def parse_args():
    p = argparse.ArgumentParser(
        description="TPS warp visualization for mammogram experiment pairs (exp1–exp5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=batch.EXPERIMENT_HELP,
    )
    p.add_argument(
        "--pack-dir",
        type=str,
        default="../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5",
    )
    p.add_argument("--out-dir", type=str, default="outputs/tps_warp")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument(
        "--run-tag-exact",
        action="store_true",
        help="Use --run-tag as-is (see batch_mammo_correspondence.resolve_run_tag)",
    )
    p.add_argument("--experiments", type=str, default="exp1,exp2,exp3,exp4,exp5")
    p.add_argument("--anchor-patient", type=str, default="11513410")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--exp12-patient", type=str, default="")
    p.add_argument("--exp3-max-sources", type=int, default=0)
    p.add_argument("--exp5-max-patients", type=int, default=0)
    p.add_argument(
        "--exp5-views",
        type=str,
        default="",
        help="Exp5 only, e.g. 62877247:L:CC,45209155:R:MLO",
    )
    p.add_argument(
        "--reuse-semcorre-dir",
        type=str,
        default="",
        help="Prior batch_mammo_correspondence run dir (reads exp5 *_chain.json; skips LDM if complete)",
    )
    p.add_argument(
        "--semcorre",
        choices=("auto", "reuse", "ldm"),
        default="auto",
        help="auto: reuse if --reuse-semcorre-dir set else LDM; reuse: batch only; ldm: always run SD",
    )
    p.add_argument(
        "--semcorre-fallback-ldm",
        action="store_true",
        help="If reuse misses a step, run LDM instead of skipping (slow)",
    )
    p.add_argument(
        "--semcorre-per-roi-point",
        action="store_true",
        help="Run SemCorre separately for each ROI corner/center (5× slower; needs LDM)",
    )
    p.add_argument("--num_steps", type=int, default=129)
    p.add_argument("--noise_level", type=int, default=-8)
    p.add_argument("--num_opt_iterations", type=int, default=3)
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=0.0023755632081200314)
    p.add_argument("--sigma", type=float, default=27.97853316316864)
    p.add_argument("--crop_percent", type=float, default=93.16549294381423)
    p.add_argument("--flip_prob", type=float, default=0.0)
    p.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--model_type", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--upsample_res", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    pack_root = Path(args.pack_dir).expanduser().resolve()
    out_base = Path(args.out_dir).expanduser().resolve()
    run_tag = batch.resolve_run_tag(args.run_tag, exact=args.run_tag_exact)
    run_root = out_base / run_tag
    print(f"Run tag: {run_tag}", flush=True)
    run_root.mkdir(parents=True, exist_ok=True)

    records = batch.load_roi_table(pack_root)
    index = batch.load_index(pack_root)
    want = {x.strip() for x in args.experiments.split(",") if x.strip()}

    tasks: list[WarpTask] = []
    exp12_pid = (args.exp12_patient or "").strip()
    exp12_set = {exp12_pid} if exp12_pid else set()

    if "exp1" in want:
        j1 = batch.build_exp1(records, pack_root, index)
        tasks.extend(pair_job_to_task(j) for j in (batch.filter_pair_jobs_for_patients(j1, exp12_set) if exp12_set else j1))
    if "exp2" in want:
        j2 = batch.build_exp2(records, pack_root, index)
        tasks.extend(pair_job_to_task(j) for j in (batch.filter_pair_jobs_for_patients(j2, exp12_set) if exp12_set else j2))
    if "exp3" in want:
        for j in batch.build_exp3(records, pack_root, args.anchor_patient, max_sources=args.exp3_max_sources):
            tasks.append(pair_job_to_task(j))
    if "exp4" in want:
        tasks.extend(pair_job_to_task(j) for j in batch.build_exp4(records, pack_root, index))

    chain_jobs: list[batch.ChainJob] = []
    if "exp5" in want:
        exp5_specs = batch.parse_exp5_view_specs(args.exp5_views)
        chain_jobs = batch.build_exp5(records, pack_root, index)
        chain_jobs = batch.filter_exp5_chains(chain_jobs, exp5_specs)
        chain_jobs = batch.limit_exp5_chains(chain_jobs, args.exp5_max_patients)

    if args.limit > 0:
        tasks = tasks[: args.limit]

    reuse_root = (
        Path(args.reuse_semcorre_dir).expanduser().resolve() if args.reuse_semcorre_dir else None
    )
    semcorre_mode = args.semcorre
    if semcorre_mode == "auto":
        semcorre_mode = "reuse" if reuse_root and reuse_root.is_dir() else "ldm"

    print(f"TPS warp: {len(tasks)} pair tasks, {len(chain_jobs)} exp5 chains")
    print(f"Output: {run_root}  |  SemCorre mode: {semcorre_mode}")

    (run_root / "run_info.json").write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiments": sorted(want),
                "n_pair_tasks": len(tasks),
                "n_exp5_chains": len(chain_jobs),
                "pack_dir": str(pack_root),
                "semcorre_per_roi_point": args.semcorre_per_roi_point,
                "reuse_semcorre_dir": str(reuse_root) if reuse_root else None,
                "semcorre_mode": semcorre_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not tasks and not chain_jobs:
        return

    need_ldm = semcorre_mode == "ldm" or args.semcorre_per_roi_point or args.semcorre_fallback_ldm
    device = args.device if torch.cuda.is_available() else "cpu"
    ldm = None
    if need_ldm:
        print(f"Loading Stable Diffusion on {device}...")
        ldm = load_ldm(device, args.model_type)
    elif semcorre_mode == "reuse":
        print(f"TPS warp: reusing SemCorre from {reuse_root} (no LDM load)", flush=True)
    hyper = dict(
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

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task.experiment} | {task.stem}")
        try:
            run_warp_task(
                ldm,
                task,
                run_root,
                device,
                hyper,
                per_roi_point_semcorre=args.semcorre_per_roi_point,
                reuse_root=reuse_root,
                semcorre_mode=semcorre_mode,
                fallback_ldm=args.semcorre_fallback_ldm,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")

    for i, chain in enumerate(chain_jobs, 1):
        print(f"\n[exp5 chain {i}/{len(chain_jobs)}] {chain.stem}")
        steps = iter_exp5_trg_steps(chain, pack_root, index)
        if not steps:
            continue
        src_path, src_xy_cur, src_box = exp5_chain_anchor(chain, pack_root)
        anchor_roi_ref = src_box
        for step_i, trg_date, trg_path in steps:
            stem = slugify(f"{chain.stem}_step{step_i:02d}_{trg_date}")
            detail = (
                f"exp5 chain patient {chain.patient_id} | {chain.laterality} {chain.view} | "
                f"step {step_i}: {Path(src_path).parent.name} → {trg_date}"
            )
            task = WarpTask(
                experiment="exp5_sequential_prior",
                experiment_label="Experiment 5: sequential backward chain (TPS step)",
                detail=detail,
                stem=stem,
                src_path=src_path,
                trg_path=trg_path,
                src_xy=src_xy_cur,
                src_gt_box=src_box if step_i == 1 else None,
                roi_size_ref_box=anchor_roi_ref,
                exp5_step=step_i,
            )
            try:
                est = run_warp_task(
                    ldm,
                    task,
                    run_root,
                    device,
                    hyper,
                    per_roi_point_semcorre=args.semcorre_per_roi_point,
                    reuse_root=reuse_root,
                    semcorre_mode=semcorre_mode,
                    fallback_ldm=args.semcorre_fallback_ldm,
                )
                if est is None:
                    break
                src_xy_cur = est
                src_path = trg_path
                src_box = None
            except Exception as exc:
                print(f"  step {step_i} FAILED: {exc}")
                break


if __name__ == "__main__":
    main()
