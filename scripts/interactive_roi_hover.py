#!/usr/bin/env python3
"""Hover an image and read EMBED-order (y, x) under the mouse.

Default: patient 45209155 2016-10-25 L CC with ROI_coords [[665, 0, 1987, 458]].

    python scripts/interactive_roi_hover.py
    python scripts/interactive_roi_hover.py --html
    python scripts/interactive_roi_hover.py --image path/to.png --roi 665,0,1987,458
"""

from __future__ import annotations

import argparse
import base64
import io
import webbrowser
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = (
    Path(r"C:\Users\paliniya\Desktop\apply\projects in progress")
    / "StableKeypointsPlus"
    / "local"
    / "data"
    / "packs"
    / "roi_overlays_cancer5"
    / "patient_45209155"
    / "2016-10-25_ROI"
    / "L_CC.png"
)
DEFAULT_ROI = (665.0, 0.0, 1987.0, 458.0)  # ymin, xmin, ymax, xmax


def _parse_roi(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",")]
    if len(parts) != 4:
        raise SystemExit("--roi must be ymin,xmin,ymax,xmax")
    ymin, xmin, ymax, xmax = (float(p) for p in parts)
    return ymin, xmin, ymax, xmax


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def write_hover_html(
    image: Image.Image,
    out_html: Path,
    *,
    roi_yx: tuple[float, float, float, float] | None,
    title: str,
    native_size: tuple[int, int] | None = None,
    max_edge: int = 1400,
) -> Path:
    """Self-contained HTML: move mouse to see y, x (native pixels)."""
    nw, nh = native_size or image.size
    preview = image
    if max(preview.size) > max_edge:
        preview = preview.copy()
        preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    roi_js = "null"
    if roi_yx is not None:
        ymin, xmin, ymax, xmax = roi_yx
        roi_js = f"{{ymin:{ymin}, xmin:{xmin}, ymax:{ymax}, xmax:{xmax}}}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0; font-family: Segoe UI, sans-serif; background: #111; color: #eee;
    }}
    header {{
      padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
      display: flex; gap: 24px; align-items: baseline; flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    #readout {{
      font-family: ui-monospace, Consolas, monospace; font-size: 18px; color: #7CFF6B;
    }}
    #hint {{ color: #aaa; font-size: 13px; }}
    #stage {{ position: relative; display: inline-block; margin: 12px; cursor: crosshair; }}
    #stage img {{ display: block; max-width: min(96vw, 1400px); height: auto; }}
    #overlay {{
      position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none;
    }}
    #tip {{
      position: absolute; transform: translate(12px, 12px);
      background: rgba(0,0,0,0.8); color: #7CFF6B; padding: 4px 8px;
      font-family: ui-monospace, Consolas, monospace; font-size: 13px;
      border: 1px solid #3a3; pointer-events: none; white-space: nowrap;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div id="readout">y = —, x = —</div>
    <div id="hint">EMBED order [y, x] in native pixels ({nw}×{nh}). Click to pin.</div>
  </header>
  <div id="stage">
    <img id="img" alt="mammogram" src="data:image/jpeg;base64,{b64}" />
    <canvas id="overlay"></canvas>
    <div id="tip" hidden></div>
  </div>
  <script>
    const nativeW = {nw}, nativeH = {nh};
    const roi = {roi_js};
    const img = document.getElementById("img");
    const canvas = document.getElementById("overlay");
    const tip = document.getElementById("tip");
    const readout = document.getElementById("readout");
    let pinned = null;

    function nativeXY(evt) {{
      const r = img.getBoundingClientRect();
      const x = (evt.clientX - r.left) * (nativeW / r.width);
      const y = (evt.clientY - r.top) * (nativeH / r.height);
      return {{ x, y }};
    }}

    function insideRoi(x, y) {{
      if (!roi) return false;
      return y >= roi.ymin && y <= roi.ymax && x >= roi.xmin && x <= roi.xmax;
    }}

    function fmt(x, y) {{
      const extra = insideRoi(x, y) ? "  · inside ROI" : "";
      return `y = ${{y.toFixed(1)}}, x = ${{x.toFixed(1)}}${{extra}}`;
    }}

    function resizeCanvas() {{
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      draw();
    }}

    function draw(cursor) {{
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const sx = w / nativeW, sy = h / nativeH;
      if (roi) {{
        ctx.strokeStyle = "#ff2a2a";
        ctx.lineWidth = 3;
        ctx.strokeRect(
          roi.xmin * sx, roi.ymin * sy,
          (roi.xmax - roi.xmin) * sx, (roi.ymax - roi.ymin) * sy
        );
      }}
      const marks = [];
      if (cursor) marks.push(cursor);
      if (pinned) marks.push(pinned);
      for (const p of marks) {{
        ctx.strokeStyle = "#7CFF6B";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(p.x * sx, 0); ctx.lineTo(p.x * sx, h);
        ctx.moveTo(0, p.y * sy); ctx.lineTo(w, p.y * sy);
        ctx.stroke();
        ctx.fillStyle = "#7CFF6B";
        ctx.beginPath();
        ctx.arc(p.x * sx, p.y * sy, 4, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    function onMove(evt) {{
      const p = nativeXY(evt);
      const text = fmt(p.x, p.y);
      readout.textContent = text;
      tip.hidden = false;
      tip.textContent = text;
      const r = img.getBoundingClientRect();
      tip.style.left = (evt.clientX - r.left) + "px";
      tip.style.top = (evt.clientY - r.top) + "px";
      draw(p);
    }}

    img.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    img.addEventListener("mousemove", onMove);
    img.addEventListener("mouseleave", () => {{
      tip.hidden = true;
      readout.textContent = pinned ? fmt(pinned.x, pinned.y) + "  (pinned)" : "y = —, x = —";
      draw(null);
    }});
    img.addEventListener("click", (evt) => {{
      pinned = nativeXY(evt);
      readout.textContent = fmt(pinned.x, pinned.y) + "  (pinned)";
    }});
    if (img.complete) resizeCanvas();
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def show_matplotlib(
    image: Image.Image,
    *,
    roi_yx: tuple[float, float, float, float] | None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(image)
    if roi_yx is not None:
        ymin, xmin, ymax, xmax = roi_yx
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=2,
                label="ROI [ymin,xmin,ymax,xmax]",
            )
        )
        ax.legend(loc="upper right")
    ax.set_title(title + "\nmove mouse — status bar and label show y, x")
    ax.set_xlabel("x (columns)")
    ax.set_ylabel("y (rows)")

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(14, 14),
        textcoords="offset points",
        color="lime",
        fontsize=11,
        bbox=dict(boxstyle="round", fc="black", ec="lime", alpha=0.8),
    )
    annot.set_visible(False)

    def format_coord(x: float, y: float) -> str:
        return f"y={y:.1f}, x={x:.1f}"

    ax.format_coord = format_coord

    def on_move(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        x, y = float(event.xdata), float(event.ydata)
        annot.xy = (x, y)
        annot.set_text(f"y = {y:.1f},  x = {x:.1f}")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive y,x hover on a mammogram.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--roi",
        default="665,0,1987,458",
        help="EMBED box ymin,xmin,ymax,xmax. Empty string to hide.",
    )
    parser.add_argument("--html", action="store_true", help="Write HTML and open in the browser.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "L_CC_hover.html",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    roi = _parse_roi(args.roi) if str(args.roi).strip() else None
    image = _load_rgb(args.image)
    title = f"{args.image.name}  {image.size[0]}×{image.size[1]}"
    if roi is not None:
        ymin, xmin, ymax, xmax = roi
        title += f"  [{ymin:g}, {xmin:g}, {ymax:g}, {xmax:g}]"
    if args.html:
        path = write_hover_html(image, args.out, roi_yx=roi, title=title)
        print(f"Wrote {path}")
        webbrowser.open(path.resolve().as_uri())
        return
    show_matplotlib(image, roi_yx=roi, title=title)


if __name__ == "__main__":
    main()
