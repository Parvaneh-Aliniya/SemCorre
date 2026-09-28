"""Run SemCorre on the paper example_images and save heatmaps with value-range colorbars.

Optimizes the token once on source_cat (paper paw query), then applies it to
target_cat / cartoon_cat / lego_cat.

From project root:
    python scripts/run_paper_example_heatmaps.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.interactive_correspondence import (  # noqa: E402
    SinglePairDataset,
    load_image_pack,
    save_correspondence_figure,
    save_heatmap_with_rois,
    sample_heatmap_at_xy,
)
from utils.optimize_token import (  # noqa: E402
    find_max_pixel_value,
    load_ldm,
    optimize_prompt,
    run_image_with_tokens_cropped,
    visualize_image_with_points,
)

# Paper custom-pair query (eval/custom_image.py): left-cat paw in Figure 3.
PAPER_SRC_XY_NORM = (0.4, 0.9)
EXAMPLE_DIR = ROOT / "example_images"
SOURCE_NAME = "source_cat.png"
TARGET_NAMES = ("target_cat.jpeg", "cartoon_cat.png", "lego_cat.png")


def _softmax_maps(layer_maps: torch.Tensor, upsample_res: int) -> torch.Tensor:
    n = layer_maps.shape[0]
    flat = torch.nn.Softmax(dim=-1)(layer_maps.reshape(n, upsample_res * upsample_res))
    return flat.reshape(n, upsample_res, upsample_res)


def optimize_contexts(ldm, src_img: torch.Tensor, src_xy: tuple[float, float], args) -> list[torch.Tensor]:
    contexts = []
    for i in range(args.num_opt_iterations):
        print(f"  optimize iteration {i + 1}/{args.num_opt_iterations}")
        contexts.append(
            optimize_prompt(
                ldm,
                src_img,
                torch.tensor(src_xy, dtype=torch.float32) / 512.0,
                num_steps=args.num_steps,
                device=args.device,
                layers=args.layers,
                lr=args.learning_rate,
                upsample_res=args.upsample_res,
                noise_level=args.noise_level,
                sigma=args.sigma,
                flip_prob=args.flip_prob,
                crop_percent=args.crop_percent,
            )
        )
    return contexts


def infer_layer_maps(ldm, image: torch.Tensor, contexts: list[torch.Tensor], args) -> torch.Tensor:
    all_maps = []
    for context in contexts:
        attn_maps, _ = run_image_with_tokens_cropped(
            ldm,
            image,
            context,
            index=0,
            upsample_res=args.upsample_res,
            noise_level=args.noise_level,
            layers=args.layers,
            device=args.device,
            crop_percent=args.crop_percent,
            num_iterations=args.num_iterations,
        )
        maps = [torch.mean(attn_maps[k], dim=0, keepdim=True) for k in range(attn_maps.shape[0])]
        all_maps.append(torch.stack(maps, dim=0))
    stacked = torch.mean(torch.stack(all_maps, dim=0), dim=0)
    return _softmax_maps(stacked, args.upsample_res)


def save_heatmap_panel(
    mean_map: torch.Tensor,
    display: torch.Tensor,
    pred_xy: tuple[float, float],
    title: str,
    save_path: Path,
    src_xy: tuple[float, float] | None = None,
):
    """Heatmap-only and photo-overlay figures, both with a labeled value-range colorbar."""
    m = mean_map.detach().cpu().float().numpy()
    vmin, vmax = float(m.min()), float(m.max())
    base = np.clip(display.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    im0 = axes[0].imshow(m, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    axes[0].scatter([pred_xy[0]], [pred_xy[1]], c="red", s=40, edgecolors="white", linewidths=0.7, zorder=5)
    if src_xy is not None:
        axes[0].scatter([src_xy[0]], [src_xy[1]], c="yellow", s=40, edgecolors="black", linewidths=0.7, zorder=5)
    axes[0].set_title("Attention heatmap")
    axes[0].set_axis_off()
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(f"min={vmin:.4g}   max={vmax:.4g}", fontsize=10)

    axes[1].imshow(base, aspect="equal")
    im1 = axes[1].imshow(m, cmap="viridis", alpha=0.55, aspect="equal", vmin=vmin, vmax=vmax)
    axes[1].scatter([pred_xy[0]], [pred_xy[1]], c="red", s=40, edgecolors="white", linewidths=0.7, zorder=5)
    if src_xy is not None:
        axes[1].scatter([src_xy[0]], [src_xy[1]], c="yellow", s=40, edgecolors="black", linewidths=0.7, zorder=5)
    axes[1].set_title("Overlay on image")
    axes[1].set_axis_off()
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label(f"min={vmin:.4g}   max={vmax:.4g}", fontsize=10)

    fig.suptitle(f"{title}\nvalue range [{vmin:.4g}, {vmax:.4g}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    src_display: torch.Tensor,
    src_xy: tuple[float, float],
    src_map: torch.Tensor,
    panels: list[dict],
    save_path: Path,
):
    n = 1 + len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.6))
    src_img = np.clip(src_display.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)
    src_m = src_map.detach().cpu().float().numpy()
    smin, smax = float(src_m.min()), float(src_m.max())

    axes[0, 0].imshow(src_img, aspect="equal")
    axes[0, 0].scatter([src_xy[0]], [src_xy[1]], c="yellow", s=45, edgecolors="black", linewidths=0.7)
    axes[0, 0].set_title("Source + query")
    axes[0, 0].set_axis_off()
    im_s = axes[1, 0].imshow(src_m, cmap="viridis", aspect="equal", vmin=smin, vmax=smax)
    axes[1, 0].scatter([src_xy[0]], [src_xy[1]], c="yellow", s=45, edgecolors="black", linewidths=0.7)
    axes[1, 0].set_title(f"Source attn  [{smin:.3g}, {smax:.3g}]")
    axes[1, 0].set_axis_off()
    fig.colorbar(im_s, ax=axes[1, 0], fraction=0.046, pad=0.03)

    for i, pan in enumerate(panels, start=1):
        img = np.clip(pan["display"].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)
        m = pan["map"].detach().cpu().float().numpy()
        vmin, vmax = float(m.min()), float(m.max())
        px, py = pan["pred"]
        axes[0, i].imshow(img, aspect="equal")
        axes[0, i].imshow(m, cmap="viridis", alpha=0.5, aspect="equal", vmin=vmin, vmax=vmax)
        axes[0, i].scatter([px], [py], c="red", s=40, edgecolors="white", linewidths=0.7)
        axes[0, i].set_title(pan["name"])
        axes[0, i].set_axis_off()
        im = axes[1, i].imshow(m, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        axes[1, i].scatter([px], [py], c="red", s=40, edgecolors="white", linewidths=0.7)
        axes[1, i].set_title(f"[{vmin:.3g}, {vmax:.3g}]")
        axes[1, i].set_axis_off()
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.03)

    fig.suptitle("Paper example images — attention heatmaps with value range", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Paper example_images heatmaps with value-range colorbars.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default="outputs/paper_examples")
    p.add_argument("--example-dir", type=str, default=str(EXAMPLE_DIR))
    p.add_argument("--num_steps", type=int, default=129)
    p.add_argument("--noise_level", type=int, default=-8)
    p.add_argument("--num_opt_iterations", type=int, default=5)
    p.add_argument("--num_iterations", type=int, default=20)
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
    example_dir = Path(args.example_dir)
    source_path = example_dir / SOURCE_NAME
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing paper source image: {source_path}")

    src_xy = (PAPER_SRC_XY_NORM[0] * 512.0, PAPER_SRC_XY_NORM[1] * 512.0)
    src_pack = load_image_pack(str(source_path))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"Device: {device}")
    print(f"Source: {source_path}  query=({src_xy[0]:.1f}, {src_xy[1]:.1f})")
    print("Loading Stable Diffusion...")
    ldm = load_ldm(device, args.model_type)

    print("Optimizing token on source (paper paw query)...")
    contexts = optimize_contexts(ldm, src_pack.clean, src_xy, args)

    print("Source attention...")
    src_maps = infer_layer_maps(ldm, src_pack.clean, contexts, args)
    src_mean = torch.mean(src_maps, dim=0)
    visualize_image_with_points(
        src_pack.clean,
        torch.tensor(src_xy),
        "source_cat_query_point",
        save_folder=str(out_dir),
    )
    visualize_image_with_points(
        src_mean[None],
        torch.tensor(src_xy),
        "source_cat_attention_mean",
        save_folder=str(out_dir),
    )
    save_heatmap_panel(
        src_mean,
        src_pack.display,
        src_xy,
        "Source attention (mean over layers)",
        out_dir / "source_cat_heatmap_with_range.png",
        src_xy=src_xy,
    )

    overview_panels = []
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "source_point_512": {"x": src_xy[0], "y": src_xy[1]},
        "device": device,
        "targets": [],
    }

    for name in TARGET_NAMES:
        target_path = example_dir / name
        if not target_path.is_file():
            print(f"Skip missing target: {target_path}")
            continue
        print(f"Target inference: {name}")
        trg_pack = load_image_pack(str(target_path))
        trg_maps = infer_layer_maps(ldm, trg_pack.clean, contexts, args)
        mean_map = torch.mean(trg_maps, dim=0)
        est = find_max_pixel_value(mean_map, img_size=512) + 0.5
        pred = (est[0].item(), est[1].item())
        hm_at = sample_heatmap_at_xy(mean_map, pred[0], pred[1])
        hm_max = float(mean_map.max().item())
        hm_min = float(mean_map.min().item())
        stem = Path(name).stem

        for k in range(trg_maps.shape[0]):
            visualize_image_with_points(
                trg_maps[k, None],
                est,
                f"{stem}_largest_loc_trg_00_{k:02d}",
                save_folder=str(out_dir),
            )
        visualize_image_with_points(
            mean_map[None],
            est,
            f"{stem}_largest_loc_trg_00_mean",
            save_folder=str(out_dir),
        )
        save_heatmap_with_rois(
            mean_map,
            out_dir / f"{stem}_target_attention_with_rois.png",
            None,
            None,
            None,
            trg_display=trg_pack.display,
        )
        save_heatmap_panel(
            mean_map,
            trg_pack.display,
            pred,
            f"{stem} attention (mean over layers)",
            out_dir / f"{stem}_heatmap_with_range.png",
        )

        dataset = SinglePairDataset(src_pack.clean, trg_pack.clean, src_xy)
        mini_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        save_correspondence_figure(
            src_pack.display,
            trg_pack.display,
            mini_batch["src_kps"][0, :, 0],
            est,
            source_name=SOURCE_NAME,
            target_name=name,
            save_path=out_dir / f"{stem}_correspondences_estimated.png",
            trg_gt_box=None,
            trg_pred_box=None,
            trg_heatmap_box=None,
            experiment_type="paper_example",
            experiment_detail=f"{SOURCE_NAME} -> {name}",
        )

        overview_panels.append({"name": stem, "display": trg_pack.display, "map": mean_map, "pred": pred})
        summary["targets"].append(
            {
                "image": str(target_path),
                "pred_512": {"x": pred[0], "y": pred[1]},
                "heatmap_min": hm_min,
                "heatmap_max": hm_max,
                "heatmap_at_pred": hm_at,
            }
        )
        print(
            f"  pred=({pred[0]:.1f}, {pred[1]:.1f})  "
            f"range=[{hm_min:.6g}, {hm_max:.6g}]  attn@pred={hm_at:.6g}"
        )

    save_overview(
        src_pack.display,
        src_xy,
        src_mean,
        overview_panels,
        out_dir / "paper_examples_heatmap_overview.png",
    )
    with open(out_dir / "paper_examples_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Heatmaps with value ranges in:\n  {out_dir}")


if __name__ == "__main__":
    main()
