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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.optimize_token import (  # noqa: E402
    find_max_pixel_value,
    load_ldm,
    optimize_prompt,
    run_image_with_tokens_cropped,
    visualize_image_with_points,
)
from utils.utils import visualie_correspondences  # noqa: E402


def load_image_tensor(path: str) -> torch.Tensor:
    """RGB CHW float [0,1], resized to 512x512 (matches eval/custom_image)."""
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), Image.BILINEAR)
    arr = np.array(image)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32) / 255.0


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
):
    """One source point, no target GT — same outputs as eval --visualize (except GT line plot)."""
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

    est = find_max_pixel_value(torch.mean(all_maps, dim=0), img_size=512) + 0.5
    est_keypoints = torch.zeros_like(mini_batch["src_kps"])
    est_keypoints[0, :, j] = est

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

    visualie_correspondences(
        mini_batch["src_img"][0],
        mini_batch["trg_img"][0],
        mini_batch["src_kps"],
        est_keypoints,
        f"{stem}_correspondences_estimated",
        correct_ids=None,
        save_folder=str(save_folder),
    )

    torch.save(
        {
            "est_keypoints": est_keypoints,
            "src_kps": mini_batch["src_kps"],
            "contexts": torch.stack(contexts),
        },
        save_folder / f"{stem}_correspondence_data.pt",
    )
    print(f"Estimated target point (512 px): x={est[0].item():.2f}, y={est[1].item():.2f}")
    return est


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
    p.add_argument("--crop_percent", type=float, default=93.16549294381423)
    p.add_argument("--flip_prob", type=float, default=0.0)
    p.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--model_type", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--upsample_res", type=int, default=512)
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

    src_t = load_image_tensor(source_path)
    trg_t = load_image_tensor(target_path)

    src_xy = pick_point_on_source(src_t, source_path)

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

    dataset = SinglePairDataset(src_t, trg_t, src_xy)
    mini_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Stable Diffusion on {device} (first run may download weights)...")
    ldm = load_ldm(device, args.model_type)

    print("Running optimization + correspondence (this can take several minutes)...\n")
    est = run_correspondence(
        ldm,
        mini_batch,
        save_folder=out_dir,
        file_stem=prefix,
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
    )

    write_run_metadata(
        out_dir, prefix, title, description, source_path, target_path, src_xy,
    )
    meta_path = out_dir / f"{prefix}_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["estimated_target_point_512"] = {"x": est[0].item(), "y": est[1].item()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Results in:\n  {out_dir}\n")
    print(f"  {prefix}_metadata.json")
    print(f"  {prefix}_README.txt")
    print(f"  {prefix}_correspondences_estimated.png")
    print(f"  {prefix}_largest_loc_trg_00_mean.png  (target attention)\n")


if __name__ == "__main__":
    main()
