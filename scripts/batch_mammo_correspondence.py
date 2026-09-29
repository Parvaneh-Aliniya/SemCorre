"""
Batch semantic correspondence on roi_overlays_cancer5 (no GUI).

Uses roi_coords.csv / .xlsx for ROI boxes and centers (native PNG pixels).
Loads **clean** PNGs (patient/<date>/...) when present; otherwise the ROI-folder PNG.

Outputs (each batch run gets a dated folder so nothing is overwritten):
  {out-dir}/{run-tag}/{experiment}/{pair-stem}/
      all PNGs (correspondence, heatmaps, optional TPS warp), .pt, pair.json for that pair

  With default --with-tps-warp: after each unwarped SemCorre run, also writes TPS figures
      ({stem}_tps_roi_warp_before_after.png and/or {stem}_tps_breast10_warp_before_after.png).
      Optional --semcorre-after-tps: second SemCorre on TPS-warped source vs same target
      ({stem}_semcorre_post_tps_roi / _breast). Disable warp with --no-with-tps-warp.

  Run folder: {out-dir}/{run-tag}/. Default tag is UTC YYYY-MM-DD_HHMMSS.
  Custom --run-tag gets _YYYY-MM-DD_HHMMSS appended so reruns do not overwrite (use --run-tag-exact to disable).

ROI: EMBED [ymin,xmin,ymax,xmax] on the PNG named in roi_coords image_path (pack index join).
  roi_center from the spreadsheet picks raw vs flipped X when PNG export differs from DICOM.
  Exp5: backward chain from roi_coords exam → strictly older exams on the timeline (one
  direction in time). No bridge ROI → newest when ROI is mid-timeline.

Experiments (--experiments comma list; aliases exp1 … exp4):

  exp1 — Within-exam view change (both directions)
      MLO → CC and CC → MLO; same patient, exam_id, laterality.
      Source: view with ROI row; target GT only if opposite view has on-tissue ROI.
      If target has no valid ROI: round-trip back to source + IoU on pair PNG.

  exp2 / exp2_cross_lateral — Cross-lateral, same view (both directions)
      L → R and R → L, same patient, same exam_id, same view (CC or MLO).
      Source: side with ROI row; target GT when opposite side has on-tissue ROI.
      If target has no valid ROI: round-trip back to source + IoU on pair PNG.

  exp3 / exp3_cross_patient — Cross-patient, matched specification
      Target (anchor): each ROI row for --anchor-patient (default all patients).
      Source: other patients' ROI images with the **same laterality and view** as that anchor.
      (No CC→MLO mixing; only patient id differs.)
      Target GT: anchor ROI on the anchor image when on-tissue.

  exp4 / exp4_prior_exam — Temporal / prior exam (same patient)
      Source: exam that has ROI (typically *_ROI folder date).
      Target: previous exam date for that patient, same laterality and view.
      Target GT: none (prior exam usually has no ROI in spreadsheet).
      When target has no ROI: step 2 round-trip (target → source) + IoU vs source ROI
      on the same {stem}_correspondences_estimated.png (cyan line, lime box, footer IoU).

  exp5 / exp5_sequential_prior — Backward chain per lat + view (ROI exam → older only)
      Starts on roi_coords exam (GT center); hops to each strictly older exam on timeline.
      Step 2+: predicted point on prior image → next older exam.
      Main overview PNG (two rows): row 1 sequential hops, row 2 same dates, direct ROI→prior.
      Also saves single-row shift/raw variants. Per-step folders: pair PNGs + reverse IoU.

Examples:
  # MLO → CC only:
  python scripts/batch_mammo_correspondence.py --experiments exp1 ...

  python scripts/batch_mammo_correspondence.py \\
    --pack-dir "../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5" \\
    --out-dir outputs/batch_experiments \\
    --experiments exp1,exp2,exp3,exp4 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from interactive_correspondence import (  # noqa: E402
    ChainPanelInfo,
    SinglePairDataset,
    box_from_center_size,
    load_ldm,
    run_correspondence,
    run_roundtrip_back_to_source,
    save_chain_overview_dual_row_figure,
    save_chain_overview_figure,
    should_run_roundtrip_back,
    slugify,
)
from utils.embed_roi import roi_for_semcorre  # noqa: E402
from utils.tps_roi_warp import (  # noqa: E402
    ControlMode,
    rgb01_to_tensor_chw,
    save_tps_warp_for_correspondence,
)

RES = 512

_RUN_TAG_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}$")


def resolve_run_tag(raw: str, *, exact: bool = False) -> str:
    """Unique batch output folder name; append UTC timestamp unless already suffixed."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    base = (raw or "").strip()
    if exact:
        return base or ts
    if not base:
        return ts
    if _RUN_TAG_TS_RE.search(base):
        return base
    return f"{base}_{ts}"


def write_center_error_summary(run_root: Path) -> Path | None:
    rows = []
    for p in sorted(run_root.rglob("*_center_error.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if not data:
            continue
        primary = data.get("vs_primary_gt") or {}
        rows.append(
            {
                "file": str(p.relative_to(run_root)).replace("\\", "/"),
                "dist_512_px": primary.get("dist_512_px"),
                "dist_scale_px": primary.get("dist_scale_px", primary.get("dist_512_px")),
                "dist_raw_px": primary.get("dist_raw_px", primary.get("dist_native_px")),
                "dist_pct_of_width": primary.get("dist_pct_of_width"),
                "dist_pct_of_diagonal": primary.get("dist_pct_of_diagonal"),
                "dist_native_px": primary.get("dist_native_px"),
                "dist_pct_of_native_width": primary.get("dist_pct_of_native_width"),
                "dist_pct_of_native_diagonal": primary.get("dist_pct_of_native_diagonal"),
                "dx_512": primary.get("dx_512"),
                "dy_512": primary.get("dy_512"),
            }
        )
    if not rows:
        return None
    out = run_root / "center_error_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"center-error summary: {len(rows)} rows → {out}", flush=True)
    return out


def pair_job_already_done(run_root: Path, job: PairJob) -> bool:
    pair_dir = run_root / job.experiment / job.stem
    return (pair_dir / f"{job.stem}_pair.json").is_file()


def chain_job_already_done(run_root: Path, job: ChainJob) -> bool:
    chain_dir = run_root / "exp5_sequential_prior" / job.stem
    chain_json = chain_dir / f"{job.stem}_chain.json"
    if not chain_json.is_file():
        return False
    try:
        data = json.loads(chain_json.read_text(encoding="utf-8"))
        return not data.get("partial", True)
    except Exception:
        return False


def _box_around_point(
    xy: tuple[float, float],
    ref_box: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if ref_box is not None:
        w = max(8.0, float(ref_box[2] - ref_box[0]))
        h = max(8.0, float(ref_box[3] - ref_box[1]))
    else:
        w, h = 48.0, 48.0
    return box_from_center_size(float(xy[0]), float(xy[1]), w, h)


def compose_gt_compare_sheet(gauss_dir: Path, roi_dir: Path, out_dir: Path) -> list[Path]:
    """Side-by-side Gaussian GT vs ROI-box GT for every matching PNG."""
    from PIL import Image, ImageDraw, ImageFont

    written: list[Path] = []
    names = sorted({p.name for p in gauss_dir.glob("*.png")} | {p.name for p in roi_dir.glob("*.png")})
    skip = ("gt_gaussian_ref.png", "gt_roi_box_mask.png")
    for name in names:
        if name.endswith(skip):
            continue
        left_p = gauss_dir / name
        right_p = roi_dir / name
        if not left_p.is_file() or not right_p.is_file():
            continue
        left = Image.open(left_p).convert("RGB")
        right = Image.open(right_p).convert("RGB")
        h = max(left.height, right.height)
        banner = 42
        canvas = Image.new("RGB", (left.width + right.width, h + banner), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        draw.text((16, 8), "Gaussian GT", fill=(20, 20, 20), font=font)
        draw.text((left.width + 16, 8), "ROI-box GT", fill=(20, 20, 20), font=font)
        canvas.paste(left, (0, banner))
        canvas.paste(right, (left.width, banner))
        dest = out_dir / name.replace(".png", "_gaussian_vs_roi.png")
        canvas.save(dest)
        written.append(dest)
    print(f"  wrote {len(written)} Gaussian vs ROI comparison figure(s) → {out_dir}", flush=True)
    return written


def run_roundtrip_if_needed(
    ldm,
    *,
    src_gt_box: tuple[float, float, float, float] | None,
    trg_gt_box: tuple[float, float, float, float] | None,
    trg_t: torch.Tensor,
    src_t: torch.Tensor,
    forward_target_kp: tuple[float, float],
    forward_src_kp: tuple[float, float],
    original_src_path: Path | str,
    original_trg_path: Path | str,
    save_folder: Path,
    file_stem: str,
    device: str,
    hyper: dict,
    experiment_type: str,
    experiment_detail: str,
    forward_trg_white_box: tuple[float, float, float, float] | None = None,
    forward_trg_pred_box: tuple[float, float, float, float] | None = None,
    src_all_gt_boxes: list | None = None,
    trg_all_gt_boxes: list | None = None,
) -> dict | None:
    src_box = src_gt_box or _box_around_point(forward_src_kp, forward_trg_white_box)
    return run_roundtrip_back_to_source(
        ldm,
        trg_t=trg_t,
        src_t=src_t,
        forward_target_kp=forward_target_kp,
        forward_src_kp=forward_src_kp,
        original_src_gt_box=src_box,
        original_src_path=str(original_src_path),
        original_trg_path=str(original_trg_path),
        save_folder=save_folder,
        file_stem=file_stem,
        device=device,
        experiment_type=experiment_type,
        experiment_detail=experiment_detail,
        forward_trg_white_box=forward_trg_white_box,
        forward_trg_pred_box=forward_trg_pred_box,
        src_all_gt_boxes=src_all_gt_boxes,
        trg_all_gt_boxes=trg_all_gt_boxes,
        **hyper,
    )


def parse_tps_modes(spec: str) -> tuple[ControlMode, ...]:
    raw = [x.strip().lower() for x in spec.split(",") if x.strip()]
    if not raw:
        return ("roi", "breast")
    modes: list[ControlMode] = []
    for x in raw:
        if x not in ("roi", "breast"):
            raise ValueError(f"Unknown TPS mode {x!r} (use roi, breast)")
        if x not in modes:
            modes.append(x)  # type: ignore[arg-type]
    return tuple(modes)  # type: ignore[return-value]


def run_tps_and_optional_semcorre_rerun(
    ldm,
    *,
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    trg_path: Path | str,
    method_src: tuple[float, float],
    method_dst: tuple[float, float],
    save_folder: Path,
    file_stem: str,
    src_gt_box,
    trg_gt_box,
    roi_size_ref_box,
    device: str,
    hyper: dict,
    experiment_type: str,
    experiment_detail: str,
    with_tps_warp: bool,
    tps_modes: tuple[ControlMode, ...],
    semcorre_after_tps: bool,
) -> None:
    if not with_tps_warp:
        return
    tps_results = save_tps_warp_for_correspondence(
        src_display=src_t,
        trg_display=trg_t,
        method_src=method_src,
        method_dst=method_dst,
        save_folder=save_folder,
        file_stem=file_stem,
        src_gt_box=src_gt_box,
        roi_size_ref_box=roi_size_ref_box,
        experiment_type=experiment_type,
        experiment_detail=experiment_detail,
        modes=tps_modes,
    )
    if not semcorre_after_tps:
        return
    trg_path = str(trg_path)
    for mode, res in tps_results.items():
        warped_t = rgb01_to_tensor_chw(res.warped_rgb)
        post_stem = f"{file_stem}_semcorre_post_tps_{mode}"
        post_detail = (
            f"{experiment_detail} | SemCorre on TPS-warped source ({mode} controls) "
            f"→ same target as unwarped run"
        )
        mini_batch = next(
            iter(
                DataLoader(
                    SinglePairDataset(warped_t, trg_t, res.warped_src_kp),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )
        run_correspondence(
            ldm,
            mini_batch,
            save_folder=save_folder,
            file_stem=post_stem,
            source_path=f"{file_stem}_tps_warped_source_{mode}",
            target_path=trg_path,
            src_display=warped_t,
            trg_display=trg_t,
            src_gt_box=src_gt_box,
            trg_gt_box=trg_gt_box,
            roi_size_ref_box=roi_size_ref_box,
            device=device,
            experiment_type=experiment_type,
            experiment_detail=post_detail,
            **hyper,
        )


@dataclass
class RoiRecord:
    patient_id: str
    exam_id: str
    exam_date: str
    laterality: str
    view: str
    image_name: str
    image_path: str
    roi_coord: tuple[float, float, float, float]  # ymin, xmin, ymax, xmax native
    roi_center: tuple[float, float]  # cx, cy native


@dataclass
class PairJob:
    experiment: str
    experiment_label: str
    experiment_detail: str
    src_path: Path
    trg_path: Path
    src_xy_512: tuple[float, float]
    src_gt_box_512: tuple[float, float, float, float] | None
    trg_gt_box_512: tuple[float, float, float, float] | None
    trg_gt_on_tissue: bool
    stem: str
    src_all_gt_512: list | None = None
    trg_all_gt_512: list | None = None


def _exp1_pairs_for_set(
    records,
    pack_root: Path,
    index,
    exp1_set: set[str],
    exam_specs: set[tuple[str, str]] | None,
) -> list[PairJob]:
    j1 = build_exp1(records, pack_root, index)
    if exp1_set:
        j1 = filter_pair_jobs_for_patients(j1, exp1_set)
    if exam_specs:
        j1 = filter_pair_jobs_for_exams(j1, exam_specs)
    return j1


def _relabel_exp6_job(job: PairJob) -> PairJob:
    return replace(
        job,
        experiment="exp6_layers_2_6",
        experiment_label="Experiment 6: DHPF layers 2–6 (combined mid stack)",
        experiment_detail=f"{job.experiment_detail} | attn layers 2-6",
    )


def _relabel_exp7_job(job: PairJob) -> PairJob:
    return replace(
        job,
        experiment="exp7_layer_ablation",
        experiment_label="Experiment 7: layer ablation (Exp1 CC↔MLO pairs)",
        experiment_detail=f"{job.experiment_detail} | layer ablation vs 7-10 baseline",
    )


@dataclass
class ChainJob:
    patient_id: str
    laterality: str
    view: str
    anchor_date: str  # sequential start = roi_coords exam date (GT anchor)
    exam_dates: list[str]  # sorted oldest → newest
    anchor_record: RoiRecord  # roi_coords row (GT box on anchor_record.image_path)
    stem: str


def roi_on_image(
    png_path: Path,
    box_yx: tuple[float, float, float, float],
    *,
    require_on_tissue: bool,
    reference_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float], bool]:
    return roi_for_semcorre(
        png_path,
        box_yx,
        require_on_tissue=require_on_tissue,
        reference_center=reference_center,
    )


def roi_on_record(
    png_path: Path,
    record: RoiRecord,
    *,
    require_on_tissue: bool,
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float], bool]:
    """Apply this CSV row's roi_coord + roi_center to the given PNG (usually record's image_path)."""
    ymin, xmin, ymax, xmax = record.roi_coord
    with Image.open(png_path) as im:
        w, h = im.size
    nw, nh = xmax - xmin, ymax - ymin
    print(
        f"  [roi] {png_path.name} native box {nw:.0f}x{nh:.0f} on PNG {w}x{h} "
        f"→ 512 box {nw * RES / w:.1f}x{nh * RES / h:.1f}",
        flush=True,
    )
    if ymax > h + 1 or xmax > w + 1:
        print(
            f"  [roi] box extends past PNG {png_path.name} "
            f"({w}x{h}); coord ymax/xmax=({ymax:.0f},{xmax:.0f}) — "
            f"likely copied from another exam/view",
            flush=True,
        )
    return roi_on_image(
        png_path,
        record.roi_coord,
        require_on_tissue=require_on_tissue,
        reference_center=record.roi_center,
    )


def load_image_chw(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((RES, RES), Image.BILINEAR)
    import numpy as np

    arr = np.array(img, dtype="float32") / 255.0
    return torch.tensor(arr.transpose(2, 0, 1))


def parse_roi_coord(s: str) -> tuple[float, float, float, float]:
    vals = ast.literal_eval(s.strip())
    return tuple(float(v) for v in vals)


def parse_roi_center(s: str) -> tuple[float, float]:
    s = str(s).strip().strip("()")
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def lookup_exam_display_meta(
    index: list[dict],
    patient_id: str,
    exam_date: str,
    laterality: str,
    view: str,
    image_path: Path,
) -> dict[str, str]:
    """Year, exam folder name, view string, and EMBED exam_id for chain overview labels."""
    lat = laterality.upper()[:1]
    view_u = view.upper()
    exam_folder = image_path.parent.name
    year = (exam_date or exam_folder)[:4]
    view_label = f"{lat} {view_u}"
    exam_id = ""
    for it in index:
        if it.get("patient_id") != patient_id or it.get("exam_date") != exam_date:
            continue
        it_lat = str(it.get("laterality") or "")[:1].upper()
        it_view = str(it.get("view") or "").upper()
        if it_lat != lat or it_view != view_u:
            continue
        exam_id = str(it.get("exam_id") or exam_id)
        if it.get("exam_folder"):
            exam_folder = str(it["exam_folder"])
    return {
        "exam_year": year,
        "exam_name": exam_folder,
        "view_label": view_label,
        "exam_id": exam_id,
    }


def resolve_exam_image(
    pack_root: Path, patient_id: str, exam_date: str, laterality: str, view: str
) -> Path | None:
    lat = laterality.upper()[:1]
    view = view.upper()
    candidates = [
        pack_root / f"patient_{patient_id}/{exam_date}/{lat}_{view}.png",
        pack_root / f"patient_{patient_id}/{exam_date}_ROI/{lat}_{view}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def patient_exam_dates(index: list[dict], patient_id: str) -> list[str]:
    dates: set[str] = set()
    for it in index:
        if it["patient_id"] == patient_id:
            dates.add(it["exam_date"])
    return sorted(dates)


def view_exam_timeline(
    pack_root: Path,
    index: list[dict],
    patient_id: str,
    laterality: str,
    view: str,
) -> list[str]:
    """Sorted exam dates for this patient + laterality + view where the PNG exists in the pack."""
    lat = laterality.upper()[:1]
    view_u = view.upper()
    dates: set[str] = set()
    for it in index:
        if it.get("patient_id") != patient_id:
            continue
        if str(it.get("laterality") or "")[:1].upper() != lat:
            continue
        if str(it.get("view") or "").upper() != view_u:
            continue
        d = str(it.get("exam_date") or "")
        if d and resolve_exam_image(pack_root, patient_id, d, lat, view_u):
            dates.add(d)
    return sorted(dates)


def parse_exp5_view_specs(raw: str) -> set[tuple[str, str, str]]:
    """Parse '62877247:L:CC,45209155:R:MLO' → {(pid, L/R, VIEW), ...}."""
    out: set[tuple[str, str, str]] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(":")]
        if len(bits) != 3:
            continue
        pid, lat, view = bits
        out.add((pid, lat.upper()[:1], view.upper()))
    return out


def filter_exp5_chains(
    chains: list[ChainJob],
    view_specs: set[tuple[str, str, str]],
) -> list[ChainJob]:
    if not view_specs:
        return chains
    return [
        c
        for c in chains
        if (c.patient_id, c.laterality.upper()[:1], c.view.upper()) in view_specs
    ]


def resolve_clean_path(pack_root: Path, image_path: str) -> Path:
    rel = Path(image_path)
    clean_rel = Path(str(rel).replace("_ROI/", "/"))
    clean = pack_root / clean_rel
    if clean.is_file():
        return clean
    overlay = pack_root / rel
    if overlay.is_file():
        return overlay
    raise FileNotFoundError(f"Missing image: {clean} or {overlay}")


def load_roi_table(pack_root: Path) -> list[RoiRecord]:
    for name in ("roi_coords.csv", "roi_coords.xlsx"):
        p = pack_root / name
        if p.is_file():
            df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
            break
    else:
        nested = pack_root / "roi_overlays_cancer5" / "roi_coords.csv"
        df = pd.read_csv(nested)

    groups: dict[tuple, list[RoiRecord]] = {}
    for _, row in df.iterrows():
        iname = str(row["image_name"])
        key = (
            str(row["patient_id"]),
            str(row["exam_id"]),
            str(row["laterality"]),
            str(row["view"]),
        )
        rec = RoiRecord(
            patient_id=str(row["patient_id"]),
            exam_id=str(row["exam_id"]),
            exam_date=str(row["exam_date"]),
            laterality=str(row["laterality"]),
            view=str(row["view"]),
            image_name=iname,
            image_path=str(row["image_path"]),
            roi_coord=parse_roi_coord(str(row["roi_coord"])),
            roi_center=parse_roi_center(str(row["roi_center"])),
        )
        groups.setdefault(key, []).append(rec)

    records: list[RoiRecord] = []
    for recs in groups.values():
        plain = [r for r in recs if not re.search(r"_\d+\.png$", r.image_name)]
        pool = plain or recs
        records.extend(pool)
    return records


def _roi_area(r: RoiRecord) -> float:
    ymin, xmin, ymax, xmax = r.roi_coord
    return max(0.0, (ymax - ymin) * (xmax - xmin))


def group_roi_records(records: list[RoiRecord]) -> dict[tuple, list[RoiRecord]]:
    out: dict[tuple, list[RoiRecord]] = {}
    for r in records:
        key = (r.patient_id, r.exam_id, r.laterality, r.view)
        out.setdefault(key, []).append(r)
    return out


def primary_records(records: list[RoiRecord]) -> list[RoiRecord]:
    return [min(recs, key=_roi_area) for recs in group_roi_records(records).values()]


def boxes_512_for_records(
    png_path: Path,
    recs: list[RoiRecord],
    *,
    require_on_tissue: bool,
) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for rec in recs:
        box, _, ok = roi_on_record(png_path, rec, require_on_tissue=require_on_tissue)
        if box is not None and (ok or not require_on_tissue):
            out.append(box)
    return out


def load_index(pack_root: Path) -> list[dict]:
    for p in (pack_root / "index.json", pack_root / "roi_overlays_cancer5" / "index.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))["items"]
    raise FileNotFoundError("index.json not found under pack-dir")


def index_lookup(index: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for it in index:
        key = (it["patient_id"], it["exam_id"], it["laterality"], it["view"])
        prev = out.get(key)
        if prev is None or ("_ROI" in it.get("exam_folder", "") and prev.get("n_roi", 0) == 0):
            out[key] = it
    return out


def roi_by_key(records: list[RoiRecord]) -> dict[tuple, RoiRecord]:
    return {k: min(v, key=_roi_area) for k, v in group_roi_records(records).items()}


def _exp1_view_change_pair(
    src: RoiRecord,
    *,
    trg_rec: RoiRecord | None,
    trg_path: Path,
    pack_root: Path,
    src_view: str,
    trg_view: str,
) -> PairJob:
    src_path = resolve_clean_path(pack_root, src.image_path)
    src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
    trg_box, _, trg_ok = (None, (0.0, 0.0), False)
    if trg_rec is not None:
        trg_box, _, trg_ok = roi_on_record(trg_path, trg_rec, require_on_tissue=True)
    direction = f"{src_view.lower()}_to_{trg_view.lower()}"
    stem = slugify(f"p{src.patient_id}_ex{src.exam_id}_{src.laterality}_{direction}")
    detail = (
        f"patient {src.patient_id} | exam {src.exam_date} | lat {src.laterality} | "
        f"source {src_view} → target {trg_view}"
    )
    if trg_rec is None:
        detail += " | target has no roi_coords row (round-trip IoU vs source ROI)"
    elif not trg_ok:
        detail += " | target GT omitted (ROI off-tissue on target PNG)"
    arrow = f"{src_view} → {trg_view}"
    return PairJob(
        f"exp1_{direction}",
        f"Experiment 1: {arrow} (same patient, exam, laterality)",
        detail,
        src_path,
        trg_path,
        src_xy,
        src_box,
        trg_box if trg_ok else None,
        trg_ok,
        stem,
    )


def build_exp1(
    records: list[RoiRecord], pack_root: Path, index: list[dict]
) -> list[PairJob]:
    """MLO↔CC within same exam; round-trip when target has no on-tissue ROI."""
    jobs: list[PairJob] = []
    groups = group_roi_records(records)
    for src_view, trg_view in (("MLO", "CC"), ("CC", "MLO")):
        for src_recs in groups.values():
            src = min(src_recs, key=_roi_area)
            if src.view != src_view:
                continue
            trg_recs = groups.get((src.patient_id, src.exam_id, src.laterality, trg_view), [])
            trg_rec = min(trg_recs, key=_roi_area) if trg_recs else None
            trg_item = find_index_item(
                index, src.patient_id, src.exam_id, src.laterality, trg_view
            )
            if trg_item is None:
                continue
            trg_path = resolve_clean_path(pack_root, trg_item["image_path"])
            job = _exp1_view_change_pair(
                src,
                trg_rec=trg_rec,
                trg_path=trg_path,
                pack_root=pack_root,
                src_view=src_view,
                trg_view=trg_view,
            )
            src_path = resolve_clean_path(pack_root, src.image_path)
            job.src_all_gt_512 = boxes_512_for_records(src_path, src_recs, require_on_tissue=False)
            if trg_recs:
                job.trg_all_gt_512 = boxes_512_for_records(
                    trg_path, trg_recs, require_on_tissue=False
                )
            jobs.append(job)
    return jobs


def find_index_item(index: list[dict], pid: str, eid: str, lat: str, view: str) -> dict | None:
    for it in index:
        if (it["patient_id"], it["exam_id"], it["laterality"], it["view"]) == (pid, eid, lat, view):
            return it
    return None


def _exp2_cross_lateral_pair(
    src: RoiRecord,
    *,
    src_lat: str,
    trg_lat: str,
    trg: RoiRecord | None,
    trg_item: dict,
    pack_root: Path,
) -> PairJob:
    src_path = resolve_clean_path(pack_root, src.image_path)
    trg_path = resolve_clean_path(pack_root, trg_item["image_path"])
    src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
    trg_box, _, trg_ok = (None, (0.0, 0.0), False)
    if trg is not None:
        trg_box, _, trg_ok = roi_on_record(trg_path, trg, require_on_tissue=True)
    stem = slugify(f"p{src.patient_id}_ex{src.exam_id}_{src.view}_{src_lat}_to_{trg_lat}")
    detail = (
        f"patient {src.patient_id} | exam {src.exam_date} | view {src.view} | "
        f"source {src_lat} → target {trg_lat}"
    )
    if trg is None:
        detail += " | target has no roi_coords row (round-trip IoU vs source ROI)"
    elif not trg_ok:
        detail += " | target GT omitted (ROI off-tissue on target PNG)"
    arrow = f"{src_lat} → {trg_lat}"
    return PairJob(
        "exp2_cross_lateral",
        f"Experiment 2: cross-lateral ({arrow}, same patient, exam, view)",
        detail,
        src_path,
        trg_path,
        src_xy,
        src_box,
        trg_box if trg_ok else None,
        trg_ok,
        stem,
    )


def build_exp2_from_set(
    records: list[RoiRecord],
    pack_root: Path,
    index: list[dict],
    review_patients: list[dict],
) -> list[PairJob]:
    """Exp2 from Experiments/lateral JSON: one L and one R per view (dates may differ)."""
    jobs: list[PairJob] = []
    seen: set[str] = set()
    for patient in review_patients:
        pid = str(patient.get("id") or "")
        if not pid:
            continue
        by_lv: dict[tuple[str, str], dict] = {}
        for ex in patient.get("exams") or []:
            for im in ex.get("images") or []:
                lat = str(im.get("laterality") or ex.get("laterality") or "")[:1].upper()
                view = str(im.get("view") or "").upper()
                if not lat or not view:
                    continue
                key = (lat, view)
                if key not in by_lv:
                    by_lv[key] = {
                        "date": str(im.get("date") or ex.get("date") or ""),
                        "laterality": lat,
                        "view": view,
                    }
        views = sorted({v for _, v in by_lv.keys()})
        for view in views:
            l_im = by_lv.get(("L", view))
            r_im = by_lv.get(("R", view))
            if not l_im or not r_im:
                continue
            for src_lat, trg_lat in (("L", "R"), ("R", "L")):
                src_im = l_im if src_lat == "L" else r_im
                trg_im = r_im if trg_lat == "R" else l_im
                src = find_roi_record(
                    records,
                    patient_id=pid,
                    exam_date=src_im["date"],
                    laterality=src_lat,
                    view=view,
                )
                trg = find_roi_record(
                    records,
                    patient_id=pid,
                    exam_date=trg_im["date"],
                    laterality=trg_lat,
                    view=view,
                )
                if src is None:
                    print(
                        f"  [exp2] skip {pid} {src_im['date']} {src_lat} {view}: no roi_coords row",
                        flush=True,
                    )
                    continue
                trg_path: Path | None = None
                if trg is not None:
                    trg_path = resolve_clean_path(pack_root, trg.image_path)
                else:
                    trg_item = next(
                        (
                            it
                            for it in index
                            if str(it.get("patient_id")) == pid
                            and str(it.get("laterality") or "")[:1].upper() == trg_lat
                            and str(it.get("view") or "").upper() == view
                            and str(it.get("exam_date") or "") == trg_im["date"]
                        ),
                        None,
                    )
                    if trg_item is None:
                        print(
                            f"  [exp2] skip {pid} → {trg_im['date']} {trg_lat} {view}: "
                            "no pack image",
                            flush=True,
                        )
                        continue
                    trg_path = resolve_clean_path(pack_root, trg_item["image_path"])
                src_path = resolve_clean_path(pack_root, src.image_path)
                src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
                trg_box, _, trg_ok = (None, (0.0, 0.0), False)
                if trg is not None:
                    trg_box, _, trg_ok = roi_on_record(trg_path, trg, require_on_tissue=True)
                same_exam = src_im["date"] == trg_im["date"]
                arrow = f"{src_lat} → {trg_lat}"
                label = (
                    f"Experiment 2: cross-lateral ({arrow}, same patient, exam, view)"
                    if same_exam
                    else f"Experiment 2: cross-lateral ({arrow}, same patient, view; review dates)"
                )
                detail = (
                    f"patient {pid} | exam {src_im['date']} | view {view} | source {src_lat} → target {trg_lat}"
                    if same_exam
                    else (
                        f"patient {pid} | view {view} | "
                        f"source {src_lat} {src_im['date']} → target {trg_lat} {trg_im['date']}"
                    )
                )
                if trg is None:
                    detail += " | target has no roi_coords row (round-trip IoU vs source ROI)"
                elif not trg_ok:
                    detail += " | target GT omitted (ROI off-tissue on target PNG)"
                stem = (
                    slugify(f"p{pid}_ex{src.exam_id}_{view}_{src_lat}_to_{trg_lat}")
                    if same_exam
                    else slugify(
                        f"p{pid}_{src_im['date']}_to_{trg_im['date']}_{view}_{src_lat}_to_{trg_lat}"
                    )
                )
                job = PairJob(
                    "exp2_cross_lateral",
                    label,
                    detail,
                    src_path,
                    trg_path,
                    src_xy,
                    src_box,
                    trg_box if trg_ok else None,
                    trg_ok,
                    stem,
                )
                if job.stem in seen:
                    continue
                seen.add(job.stem)
                src_recs = [
                    r
                    for r in records
                    if r.patient_id == pid
                    and r.exam_date == src_im["date"]
                    and r.laterality.upper()[:1] == src_lat
                    and r.view.upper() == view
                ]
                job.src_all_gt_512 = boxes_512_for_records(
                    src_path, src_recs, require_on_tissue=False
                )
                if trg is not None:
                    trg_recs = [
                        r
                        for r in records
                        if r.patient_id == pid
                        and r.exam_date == trg_im["date"]
                        and r.laterality.upper()[:1] == trg_lat
                        and r.view.upper() == view
                    ]
                    job.trg_all_gt_512 = boxes_512_for_records(
                        trg_path, trg_recs, require_on_tissue=False
                    )
                jobs.append(job)
    return jobs


def build_exp2(records: list[RoiRecord], pack_root: Path, index: list[dict]) -> list[PairJob]:
    """L↔R for each laterality+view; opposite lateral PNG from pack index."""
    jobs: list[PairJob] = []
    groups = group_roi_records(records)
    seen: set[str] = set()
    for src_lat, trg_lat in (("L", "R"), ("R", "L")):
        for src_recs in groups.values():
            src = min(src_recs, key=_roi_area)
            if src.laterality.upper()[:1] != src_lat:
                continue
            trg_recs = groups.get((src.patient_id, src.exam_id, trg_lat, src.view), [])
            trg = min(trg_recs, key=_roi_area) if trg_recs else None
            trg_item = find_index_item(index, src.patient_id, src.exam_id, trg_lat, src.view)
            if trg_item is None:
                continue
            job = _exp2_cross_lateral_pair(
                src,
                src_lat=src_lat,
                trg_lat=trg_lat,
                trg=trg,
                trg_item=trg_item,
                pack_root=pack_root,
            )
            if job.stem in seen:
                continue
            seen.add(job.stem)
            src_path = resolve_clean_path(pack_root, src.image_path)
            trg_path = resolve_clean_path(pack_root, trg_item["image_path"])
            job.src_all_gt_512 = boxes_512_for_records(src_path, src_recs, require_on_tissue=False)
            if trg_recs:
                job.trg_all_gt_512 = boxes_512_for_records(
                    trg_path, trg_recs, require_on_tissue=False
                )
            jobs.append(job)
    return jobs


def build_exp3(
    records: list[RoiRecord],
    pack_root: Path,
    anchor_patient: str,
    max_sources: int = 0,
    source_patient: str = "",
) -> list[PairJob]:
    """Cross-patient pairs with matched laterality + view (CC→CC, MLO→MLO, same L/R)."""
    jobs: list[PairJob] = []
    if not (anchor_patient or "").strip() or str(anchor_patient).strip().lower() == "all":
        anchor_records = list(records)
    else:
        anchor_records = [r for r in records if r.patient_id == anchor_patient]
        if not anchor_records:
            anchor_records = [records[0]]
    source_pid = (source_patient or "").strip()

    for anchor in anchor_records:
        trg_path = resolve_clean_path(pack_root, anchor.image_path)
        trg_box, _, trg_ok = roi_on_record(trg_path, anchor, require_on_tissue=True)

        for src in records:
            if src.patient_id == anchor.patient_id:
                continue
            if source_pid and src.patient_id != source_pid:
                continue
            if src.view != anchor.view or src.laterality != anchor.laterality:
                continue
            src_path = resolve_clean_path(pack_root, src.image_path)
            src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
            spec = f"{src.laterality}_{src.view}"
            stem = slugify(
                f"src_p{src.patient_id}_{src.exam_date}_trg_p{anchor.patient_id}_{anchor.exam_date}_{spec}"
            )
            detail = (
                f"TARGET anchor patient {anchor.patient_id} exam {anchor.exam_date} "
                f"{anchor.laterality} {anchor.view} | "
                f"SOURCE patient {src.patient_id} exam {src.exam_date} "
                f"{src.laterality} {src.view} (matched spec)"
            )
            jobs.append(
                PairJob(
                    "exp3_cross_patient",
                    "Experiment 3: cross-patient (same lat + view, anchor target)",
                    detail,
                    src_path,
                    trg_path,
                    src_xy,
                    src_box,
                    trg_box if trg_ok else None,
                    trg_ok,
                    stem,
                )
            )
    if max_sources > 0:
        jobs = jobs[:max_sources]
    return jobs


def exam_native_roi(
    records: list[RoiRecord] | None,
    pack_root: Path,
    patient_id: str,
    exam_date: str,
    laterality: str,
    view: str,
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float] | None]:
    """Own ROI on this exam/lat/view, if the pack has one."""
    if not records:
        return None, None
    rec = find_roi_record(
        records,
        patient_id=patient_id,
        exam_date=exam_date,
        laterality=laterality,
        view=view,
    )
    if rec is None:
        return None, None
    try:
        path = resolve_clean_path(pack_root, rec.image_path)
        box, xy, _ = roi_on_record(path, rec, require_on_tissue=False)
        return box, xy
    except Exception:
        return None, None


def find_roi_record(
    records: list[RoiRecord],
    *,
    patient_id: str,
    exam_date: str,
    laterality: str,
    view: str,
) -> RoiRecord | None:
    lat = laterality.upper()[:1]
    view_u = view.upper()
    hits = [
        r
        for r in records
        if r.patient_id == patient_id
        and r.exam_date == exam_date
        and r.laterality.upper()[:1] == lat
        and r.view.upper() == view_u
    ]
    if not hits:
        return None
    return min(hits, key=_roi_area)


def build_exp3_from_set(
    records: list[RoiRecord],
    pack_root: Path,
    groups: list[dict],
) -> list[PairJob]:
    """Source patients in a folder → that folder's *_target, same view (CC→CC, MLO→MLO)."""
    jobs: list[PairJob] = []
    seen: set[str] = set()
    for group in groups:
        bucket = str(group.get("bucket") or "")
        pairs = group.get("pairs") or []
        for pair in pairs:
            src = find_roi_record(
                records,
                patient_id=str(pair["source_id"]),
                exam_date=str(pair["source_date"]),
                laterality=str(pair["source_laterality"]),
                view=str(pair["source_view"]),
            )
            trg = find_roi_record(
                records,
                patient_id=str(pair["target_id"]),
                exam_date=str(pair["target_date"]),
                laterality=str(pair["target_laterality"]),
                view=str(pair["target_view"]),
            )
            if src is None or trg is None:
                print(
                    f"  [exp3] skip {pair.get('source_id')} {pair.get('source_date')} "
                    f"{pair.get('source_laterality')} {pair.get('source_view')} → "
                    f"{pair.get('target_id')} {pair.get('target_date')} "
                    f"{pair.get('target_laterality')} {pair.get('target_view')} "
                    f"(src={'ok' if src else 'missing'} trg={'ok' if trg else 'missing'})",
                    flush=True,
                )
                continue
            src_path = resolve_clean_path(pack_root, src.image_path)
            trg_path = resolve_clean_path(pack_root, trg.image_path)
            src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
            trg_box, _, trg_ok = roi_on_record(trg_path, trg, require_on_tissue=True)
            spec = f"{src.laterality}_{src.view}_to_{trg.laterality}_{trg.view}"
            stem = slugify(
                f"src_p{src.patient_id}_{src.exam_date}_trg_p{trg.patient_id}_{trg.exam_date}_{spec}"
            )
            if stem in seen:
                continue
            seen.add(stem)
            detail = (
                f"TARGET {trg.patient_id} {trg.exam_date} {trg.laterality} {trg.view} | "
                f"SOURCE {src.patient_id} {src.exam_date} {src.laterality} {src.view} | "
                f"bucket {bucket or 'set'}"
            )
            jobs.append(
                PairJob(
                    "exp3_cross_patient",
                    "Experiment 3: cross-patient (review set, same view → *_target)",
                    detail,
                    src_path,
                    trg_path,
                    src_xy,
                    src_box,
                    trg_box if trg_ok else None,
                    trg_ok,
                    stem,
                )
            )
    return jobs


def _job_patient_id(job: PairJob) -> str | None:
    m = re.search(r"p(\d+)_", job.stem)
    return m.group(1) if m else None


def parse_exam_specs(raw_list: list[str] | None) -> set[tuple[str, str]]:
    """Parse repeated --exam pid:YYYY-MM-DD."""
    out: set[tuple[str, str]] = set()
    for raw in raw_list or []:
        part = (raw or "").strip()
        if ":" not in part:
            continue
        pid, day = part.split(":", 1)
        pid, day = pid.strip(), day.strip()
        if pid and day:
            out.add((pid, day))
    return out


def load_set_json(path: str | Path) -> dict:
    """Review-folder set: exam_specs, patient_ids, optional exp5 view specs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    exams = parse_exam_specs(data.get("exam_specs") or [])
    patients = {str(p) for p in (data.get("patient_ids") or [])}
    views: set[tuple[str, str, str]] = set()
    for rec in data.get("patients") or []:
        pid = str(rec.get("id") or "")
        if pid:
            patients.add(pid)
        for ex in rec.get("exams") or []:
            day = str(ex.get("date") or "")
            lat = str(ex.get("laterality") or "")[:1].upper()
            if pid and day:
                exams.add((pid, day))
            for im in ex.get("images") or []:
                view = str(im.get("view") or "").upper()
                im_lat = str(im.get("laterality") or lat)[:1].upper()
                if pid and im_lat and view:
                    views.add((pid, im_lat, view))
        for ch in rec.get("chains") or []:
            la = str(ch.get("laterality") or "")[:1].upper()
            vw = str(ch.get("view") or "").upper()
            if pid and la and vw:
                views.add((pid, la, vw))
    for ch in data.get("chains") or []:
        pid = str(ch.get("patient_id") or ch.get("id") or "")
        la = str(ch.get("laterality") or "")[:1].upper()
        vw = str(ch.get("view") or "").upper()
        if pid and la and vw:
            views.add((pid, la, vw))
            patients.add(pid)
    return {
        "experiment": str(data.get("experiment") or ""),
        "exam_specs": exams,
        "patient_ids": patients,
        "exp5_views": views,
        "exp3_groups": data.get("groups") or [],
        "review_patients": data.get("patients") or [],
        "pair_stems": [str(s).strip() for s in (data.get("pair_stems") or []) if str(s).strip()],
        "raw": data,
    }


def filter_pair_jobs_by_stems(jobs: list[PairJob], stems: set[str]) -> list[PairJob]:
    if not stems:
        return jobs
    out = [j for j in jobs if j.stem in stems]
    missing = stems - {j.stem for j in out}
    if missing:
        print("WARNING: pair stem(s) not in planned jobs:", flush=True)
        for s in sorted(missing):
            print(f"  missing: {s}", flush=True)
    return out


def filter_pair_jobs_for_exams(
    jobs: list[PairJob], exam_specs: set[tuple[str, str]]
) -> list[PairJob]:
    if not exam_specs:
        return jobs
    out: list[PairJob] = []
    for j in jobs:
        pid = _job_patient_id(j)
        src = str(j.src_path).replace("\\", "/")
        for p, day in exam_specs:
            if pid == p and day in src:
                out.append(j)
                break
    return out


def filter_pair_jobs_for_patients(jobs: list[PairJob], patient_ids: set[str]) -> list[PairJob]:
    if not patient_ids:
        return jobs
    out: list[PairJob] = []
    for j in jobs:
        pid = _job_patient_id(j)
        if pid is None:
            for p in patient_ids:
                if p in j.stem or p in str(j.src_path):
                    out.append(j)
                    break
        elif pid in patient_ids:
            out.append(j)
    return out


def limit_exp5_chains(chains: list[ChainJob], max_patients: int) -> list[ChainJob]:
    if max_patients <= 0:
        return chains
    allowed: list[str] = []
    for c in chains:
        if c.patient_id not in allowed and len(allowed) < max_patients:
            allowed.append(c.patient_id)
    allowed_set = set(allowed)
    return [c for c in chains if c.patient_id in allowed_set]


def build_exp4(records: list[RoiRecord], pack_root: Path, index: list[dict]) -> list[PairJob]:
    jobs: list[PairJob] = []
    by_patient: dict[str, set[str]] = {}
    for it in index:
        if "_ROI" in it.get("exam_folder", ""):
            continue
        by_patient.setdefault(it["patient_id"], set()).add(it["exam_date"])
    for r in records:
        by_patient.setdefault(r.patient_id, set()).add(r.exam_date)
    by_patient_sorted = {p: sorted(d) for p, d in by_patient.items()}

    for src in records:
        dates = by_patient_sorted.get(src.patient_id, [])
        if src.exam_date not in dates:
            dates = sorted(set(dates) | {src.exam_date})
        try:
            idx = dates.index(src.exam_date)
        except ValueError:
            continue
        if idx == 0:
            continue
        prior_date = dates[idx - 1]
        trg_rel = f"patient_{src.patient_id}/{prior_date}/{src.laterality}_{src.view}.png"
        trg_path = pack_root / trg_rel
        if not trg_path.is_file():
            continue
        src_path = resolve_clean_path(pack_root, src.image_path)
        src_box, src_xy, _ = roi_on_record(src_path, src, require_on_tissue=False)
        stem = slugify(f"p{src.patient_id}_{src.exam_date}_to_{prior_date}_{src.laterality}_{src.view}")
        detail = (
            f"patient {src.patient_id} | source ROI exam {src.exam_date} {src.laterality} {src.view} → "
            f"target prior exam {prior_date} (no target GT ROI)"
        )
        jobs.append(
            PairJob(
                "exp4_prior_exam",
                "Experiment 4: prior exam (temporal, target usually without ROI)",
                detail,
                src_path,
                trg_path,
                src_xy,
                src_box,
                None,
                False,
                stem,
            )
        )
    return jobs


def build_exp5(
    records: list[RoiRecord], pack_root: Path, index: list[dict]
) -> list[ChainJob]:
    """One backward chain per (patient, laterality, view) that has ROI metadata.

    Chain runs ROI exam → strictly older exams on the pack timeline (one direction).
    GT ROI from anchor_record (latest roi_coords row for that lat+view on timeline).
    """
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[RoiRecord]] = defaultdict(list)
    for r in records:
        lat = r.laterality.upper()[:1]
        view = r.view.upper()
        groups[(r.patient_id, lat, view)].append(r)

    jobs: list[ChainJob] = []
    seen: set[tuple[str, str, str]] = set()
    for (pid, lat, view), recs in sorted(groups.items()):
        if (pid, lat, view) in seen:
            continue
        seen.add((pid, lat, view))
        timeline = view_exam_timeline(pack_root, index, pid, lat, view)
        if len(timeline) < 2:
            continue
        anchor_record = max(recs, key=lambda r: r.exam_date)
        if anchor_record.exam_date not in timeline:
            print(
                f"  [exp5] skip {pid} {lat} {view}: ROI exam {anchor_record.exam_date} "
                f"not in pack timeline",
                flush=True,
            )
            continue
        roi_exam_date = anchor_record.exam_date
        roi_idx = timeline.index(roi_exam_date)
        if roi_idx <= 0:
            continue
        if len(timeline) < 2:
            continue
        stem = slugify(f"p{pid}_{lat}_{view}_chain_roi_{roi_exam_date}")
        jobs.append(
            ChainJob(
                patient_id=pid,
                laterality=lat,
                view=view,
                anchor_date=roi_exam_date,
                exam_dates=timeline,
                anchor_record=anchor_record,
                stem=stem,
            )
        )
    return jobs


def write_exp5_overview_pngs(
    *,
    chain_dir: Path,
    stem: str,
    panels: list,
    anchor_panels: list,
    label: str,
    detail: str,
    step_meta: list[dict],
    roi_size_ref_box=None,
) -> dict[str, str]:
    """Write exp5 strip figures after each hop so they exist before the full chain ends."""
    step_lines = [
        f"Step {m['step']}: {m['source_exam']} → {m['target_exam']}  "
        f"kp=({m['target_kp_512']['x']:.0f},{m['target_kp_512']['y']:.0f})"
        for m in step_meta
    ]
    paths: dict[str, str] = {}
    overview = chain_dir / f"{stem}_chain_overview.png"
    if len(anchor_panels) == len(panels) and len(panels) > 0:
        try:
            save_chain_overview_dual_row_figure(
                panels,
                anchor_panels,
                overview,
                experiment_type=label,
                experiment_detail=detail,
                step_lines=step_lines,
                roi_size_ref_box=roi_size_ref_box,
            )
            paths["overview_png"] = str(overview)
            print(f"  wrote {overview.name} (2 rows × {len(panels)} panels, original xy)", flush=True)
        except Exception as exc:
            print(f"  FAILED dual-row overview: {exc}", flush=True)
    else:
        print(
            f"  skip dual-row overview (row1={len(panels)} panels, row2={len(anchor_panels)})",
            flush=True,
        )
    overview_single = chain_dir / f"{stem}_chain_overview_single_row.png"
    try:
        save_chain_overview_figure(
            panels,
            overview_single,
            experiment_type=label,
            experiment_detail=detail,
            step_lines=step_lines,
            roi_size_ref_box=roi_size_ref_box,
        )
        paths["overview_single_row_png"] = str(overview_single)
    except Exception as exc:
        print(f"  FAILED single-row overview: {exc}", flush=True)
    return paths


def rebuild_exp5_overviews_from_chain_json(
    chain_json: Path,
    pack_root: Path | None = None,
) -> dict[str, str]:
    """Rebuild the multi-exam strip from an existing *_chain.json (no SemCorre / no LDM)."""
    data = json.loads(chain_json.read_text(encoding="utf-8"))
    steps = data.get("steps") or []
    if not steps:
        print(f"  skip {chain_json.name}: no steps", flush=True)
        return {}
    pid = str(data.get("patient_id") or "")
    lat = str(data.get("laterality") or "")[:1].upper()
    view = str(data.get("view") or "").upper()
    anchor_date = str(data.get("anchor_date") or steps[0].get("source_exam") or "")
    roi_exam_date = str(data.get("roi_exam_date") or anchor_date)
    chain_dir = chain_json.parent
    name = chain_json.name
    stem = name[: -len("_chain.json")] if name.endswith("_chain.json") else chain_json.stem

    src_box = None
    roi_xy = None
    rec = None
    if pack_root is not None:
        try:
            records = load_roi_table(pack_root)
            rec = next(
                (
                    r
                    for r in records
                    if r.patient_id == pid
                    and str(r.laterality).upper()[:1] == lat
                    and str(r.view).upper() == view
                ),
                None,
            )
            if rec is not None:
                src_path0 = Path(steps[0]["source_path"])
                src_box, roi_xy, _ = roi_on_record(src_path0, rec, require_on_tissue=False)
        except Exception as exc:
            print(f"  ROI lookup failed ({exc}); drawing keypoints only", flush=True)

    def _box_at(kp: tuple[float, float] | None):
        if src_box is None or kp is None:
            return None
        return box_from_center_size(kp[0], kp[1], src_box[2] - src_box[0], src_box[3] - src_box[1])

    def _panel(exam_date: str, path: Path, kp, *, role: str, step_index: int = 0, hm=None, hm_max=None, gt=None, white=None):
        return ChainPanelInfo(
            exam_date=exam_date,
            title=f"{exam_date}\n{path.name}",
            display=load_image_chw(path),
            keypoint=kp,
            gt_box=gt,
            white_box=white,
            role=role,
            exam_year=exam_date[:4],
            exam_name=path.parent.name,
            view_label=f"{lat} {view}",
            step_index=step_index,
            heatmap_score_at_kp=hm,
            heatmap_map_max=hm_max,
        )

    src0 = Path(steps[0]["source_path"])
    sk0 = steps[0].get("source_kp_512") or {}
    start_kp = (
        (float(sk0["x"]), float(sk0["y"]))
        if "x" in sk0
        else roi_xy
    )
    roi_panel_path = src0
    if pack_root is not None and rec is not None:
        try:
            roi_panel_path = resolve_clean_path(pack_root, rec.image_path)
        except Exception:
            roi_panel_path = src0
    panels = [
        _panel(
            anchor_date or steps[0]["source_exam"],
            src0,
            start_kp,
            role="chain anchor",
            gt=src_box if Path(src0).resolve() == Path(roi_panel_path).resolve() else None,
        )
    ]
    anchor_panels = [
        _panel(
            roi_exam_date,
            roi_panel_path,
            roi_xy or start_kp,
            role="ROI exam (GT anchor)",
            gt=src_box,
        )
    ]
    for st in steps:
        trg = Path(st["target_path"])
        tk = st.get("target_kp_512") or {}
        seq_kp = (float(tk["x"]), float(tk["y"])) if "x" in tk else None
        ad = st.get("anchor_direct_target_kp_512") or st.get("roi_gt_forward_target_kp_512") or tk
        ad_kp = (float(ad["x"]), float(ad["y"])) if ad and "x" in ad else seq_kp
        hm = st.get("heatmap_score_at_pred")
        hm_max = st.get("heatmap_map_max")
        panels.append(
            _panel(
                str(st["target_exam"]),
                trg,
                seq_kp,
                role=f"step {st['step']} target",
                step_index=int(st["step"]),
                hm=hm,
                hm_max=hm_max,
                white=_box_at(seq_kp),
            )
        )
        anchor_panels.append(
            _panel(
                str(st["target_exam"]),
                trg,
                ad_kp,
                role=f"ROI exam → {st['target_exam']}",
                step_index=int(st["step"]),
                hm=st.get("anchor_direct_heatmap_score_at_pred") or hm,
                hm_max=st.get("anchor_direct_heatmap_map_max") or hm_max,
                white=_box_at(ad_kp),
            )
        )

    label = "Experiment 5: sequential backward chain (ROI anchor → older exams)"
    detail = (
        f"patient {pid} | {lat} {view} | "
        f"chain from {anchor_date} (ROI exam {roi_exam_date}) → {len(steps)} older exam(s)"
    )
    paths = write_exp5_overview_pngs(
        chain_dir=chain_dir,
        stem=stem,
        panels=panels,
        anchor_panels=anchor_panels,
        label=label,
        detail=detail,
        step_meta=steps,
        roi_size_ref_box=src_box,
    )
    merged = dict(data)
    merged.update(paths)
    merged["partial"] = False
    chain_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return paths


def run_chain_job(
    ldm,
    job: ChainJob,
    run_root: Path,
    pack_root: Path,
    index: list[dict],
    device: str,
    hyper: dict,
    *,
    with_tps_warp: bool = True,
    tps_modes: tuple[ControlMode, ...] = ("roi", "breast"),
    semcorre_after_tps: bool = False,
    exp5_max_steps: int = 0,
    chain_dir: Path | None = None,
    records: list[RoiRecord] | None = None,
    source_mode: str = "transfer",
) -> None:
    experiment = "exp5_sequential_prior"
    label = "Experiment 5: sequential backward chain (ROI anchor → older exams)"
    chain_dir = chain_dir or (run_root / experiment / job.stem)
    chain_dir.mkdir(parents=True, exist_ok=True)

    dates = view_exam_timeline(
        pack_root, index, job.patient_id, job.laterality, job.view
    )
    if not dates:
        dates = list(job.exam_dates)
    timeline_newest = dates[-1]
    roi_exam_date = job.anchor_record.exam_date
    seq_start_date = job.anchor_date
    if seq_start_date not in dates:
        seq_start_date = roi_exam_date
    if roi_exam_date not in dates:
        print(f"  [exp5] skip: ROI exam {roi_exam_date} not on timeline", flush=True)
        return
    if seq_start_date != roi_exam_date:
        print(
            f"  [exp5] note: job anchor_date {seq_start_date} != ROI exam {roi_exam_date}; "
            f"using ROI exam as sequential start",
            flush=True,
        )
        seq_start_date = roi_exam_date
    seq_idx = dates.index(seq_start_date)
    if seq_idx <= 0:
        print(f"  [exp5] skip: no exams older than ROI exam {seq_start_date}", flush=True)
        return
    n_older = seq_idx
    older_dates = list(reversed(dates[:seq_idx]))
    n_run = n_older if exp5_max_steps <= 0 else min(n_older, exp5_max_steps)
    hop_dates = older_dates[:n_run]
    print(
        f"  exp5 chain: start ROI exam {seq_start_date} | timeline newest {timeline_newest} | "
        f"{len(dates)} exams on timeline, {n_older} older than ROI, "
        f"running {len(hop_dates)} hop(s)"
        f" (max_steps={exp5_max_steps if exp5_max_steps > 0 else 'all'})",
        flush=True,
    )
    for i, d in enumerate(hop_dates, 1):
        src_d = seq_start_date if i == 1 else hop_dates[i - 2]
        print(f"    hop {i}: {src_d} → {d}", flush=True)
    if exp5_max_steps > 0 and n_run < n_older:
        print(
            f"  exp5: --exp5-max-steps {exp5_max_steps} limits this run "
            f"(pass 0 for all {n_older} hops)",
            flush=True,
        )

    source_mode = (source_mode or "transfer").strip().lower()
    if source_mode not in {"transfer", "own_roi"}:
        source_mode = "transfer"
    print(f"  exp5 source_mode={source_mode}", flush=True)

    roi_path = resolve_clean_path(pack_root, job.anchor_record.image_path)
    roi_box, roi_xy, _ = roi_on_record(roi_path, job.anchor_record, require_on_tissue=False)
    roi_t = load_image_chw(roi_path)

    def _own_roi(day: str):
        return exam_native_roi(
            records, pack_root, job.patient_id, day, job.laterality, job.view
        )

    roi_own_box, roi_own_xy = _own_roi(roi_exam_date)
    if source_mode == "own_roi" and roi_own_xy is not None:
        seq_start_kp = roi_own_xy
        seq_start_gt = roi_own_box
        print(f"  exp5 own_roi: using native ROI on {roi_exam_date}", flush=True)
    else:
        seq_start_kp = roi_xy
        seq_start_gt = roi_box

    roi_meta = lookup_exam_display_meta(
        index,
        job.patient_id,
        roi_exam_date,
        job.laterality,
        job.view,
        roi_path,
    )
    seq_role = "ROI exam (GT anchor; sequential start)"
    panels: list[ChainPanelInfo] = [
        ChainPanelInfo(
            exam_date=roi_exam_date,
            title=f"{roi_exam_date}\n{roi_path.name}",
            display=roi_t,
            keypoint=seq_start_kp,
            gt_box=seq_start_gt,
            role=seq_role,
            exam_year=roi_meta["exam_year"],
            exam_name=roi_meta["exam_name"],
            view_label=roi_meta["view_label"],
            exam_id=roi_meta["exam_id"],
        )
    ]
    anchor_panels: list[ChainPanelInfo] = [
        ChainPanelInfo(
            exam_date=roi_exam_date,
            title=f"{roi_exam_date}\n{roi_path.name}",
            display=roi_t,
            keypoint=roi_xy,
            gt_box=roi_box,
            role="ROI exam (GT anchor)",
            exam_year=roi_meta["exam_year"],
            exam_name=roi_meta["exam_name"],
            view_label=roi_meta["view_label"],
            exam_id=roi_meta["exam_id"],
        )
    ]
    src_path = roi_path
    src_t = roi_t
    src_xy_cur = seq_start_kp
    src_own_gt = seq_start_gt
    bridge_meta = None
    anchor_size_ref = roi_box
    step_meta: list[dict] = []

    for step, trg_date in enumerate(hop_dates, start=1):
        trg_path = resolve_exam_image(
            pack_root, job.patient_id, trg_date, job.laterality, job.view
        )
        if trg_path is None:
            print(f"  [exp5] skip step {step}: no image for {trg_date}", flush=True)
            continue
        trg_t = load_image_chw(trg_path)
        trg_own_box, trg_own_xy = _own_roi(trg_date)
        step_dir = chain_dir / f"step_{step:02d}_{seq_start_date}_to_{trg_date}"
        step_dir.mkdir(parents=True, exist_ok=True)
        stem = slugify(f"step{step:02d}_{trg_date}")
        mini_batch = next(
            iter(
                DataLoader(
                    SinglePairDataset(src_t, trg_t, src_xy_cur),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )
        detail = (
            f"patient {job.patient_id} | {job.laterality} {job.view} | "
            f"step {step}: {Path(src_path).parent.name} → {trg_date}"
        )
        (
            est,
            roi_iou_pred,
            roi_iou_heatmap,
            _trg_gt,
            trg_pred_box,
            trg_heatmap_box,
            trg_peak_box,
            roi_iou_peak,
            trg_white_box,
            hm_at_pred,
            hm_max,
        ) = run_correspondence(
            ldm,
            mini_batch,
            save_folder=step_dir,
            file_stem=stem,
            source_path=str(src_path),
            target_path=str(trg_path),
            src_display=src_t,
            trg_display=trg_t,
            src_gt_box=src_own_gt or (anchor_size_ref if step == 1 else None),
            trg_gt_box=trg_own_box,
            roi_size_ref_box=anchor_size_ref,
            device=device,
            experiment_type=label,
            experiment_detail=detail,
            **hyper,
        )
        tx, ty = est[0].item(), est[1].item()
        step_src_xy = (float(src_xy_cur[0]), float(src_xy_cur[1]))

        hop_is_roi_source = Path(src_path).resolve() == Path(roi_path).resolve()
        sequential_rt = run_roundtrip_if_needed(
            ldm,
            src_gt_box=roi_box if hop_is_roi_source else None,
            trg_gt_box=None,
            trg_t=trg_t,
            src_t=src_t,
            forward_target_kp=(tx, ty),
            forward_src_kp=step_src_xy,
            original_src_path=src_path,
            original_trg_path=trg_path,
            save_folder=step_dir,
            file_stem=stem,
            device=device,
            hyper=hyper,
            experiment_type=label,
            experiment_detail=detail,
            forward_trg_white_box=trg_white_box,
            forward_trg_pred_box=trg_pred_box,
        )
        bidirectional = sequential_rt
        tx_roi, ty_roi = tx, ty
        trg_pred_box_a = trg_pred_box
        trg_heatmap_box_a = trg_heatmap_box
        trg_peak_box_a = trg_peak_box
        trg_white_box_a = trg_white_box
        hm_at_pred_a = hm_at_pred
        hm_max_a = hm_max
        if not hop_is_roi_source:
            roi_pair_stem = slugify(f"roi_gt_to_{trg_date}")
            roi_pair_detail = (
                f"patient {job.patient_id} | {job.laterality} {job.view} | "
                f"ROI source {roi_exam_date} → prior {trg_date} (steps 1–2 bidirectional)"
            )
            roi_batch = next(
                iter(
                    DataLoader(
                        SinglePairDataset(roi_t, trg_t, roi_xy),
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                    )
                )
            )
            (
                est_roi,
                _,
                _,
                _,
                trg_pred_box_a,
                trg_heatmap_box_a,
                trg_peak_box_a,
                _,
                trg_white_roi,
                hm_at_pred_a,
                hm_max_a,
            ) = run_correspondence(
                ldm,
                roi_batch,
                save_folder=step_dir,
                file_stem=roi_pair_stem,
                source_path=str(roi_path),
                target_path=str(trg_path),
                src_display=roi_t,
                trg_display=trg_t,
                src_gt_box=roi_box,
                trg_gt_box=None,
                roi_size_ref_box=anchor_size_ref,
                device=device,
                experiment_type=label,
                experiment_detail=roi_pair_detail,
                **hyper,
            )
            tx_roi, ty_roi = est_roi[0].item(), est_roi[1].item()
            trg_white_box_a = trg_white_roi
            bidirectional = run_roundtrip_if_needed(
                ldm,
                src_gt_box=roi_box,
                trg_gt_box=None,
                trg_t=trg_t,
                src_t=roi_t,
                forward_target_kp=(tx_roi, ty_roi),
                forward_src_kp=roi_xy,
                original_src_path=roi_path,
                original_trg_path=trg_path,
                save_folder=step_dir,
                file_stem=roi_pair_stem,
                device=device,
                hyper=hyper,
                experiment_type=label,
                experiment_detail=roi_pair_detail,
                forward_trg_white_box=trg_white_roi,
                forward_trg_pred_box=trg_pred_box_a,
            )

        run_tps_and_optional_semcorre_rerun(
            ldm,
            src_t=src_t,
            trg_t=trg_t,
            trg_path=trg_path,
            method_src=step_src_xy,
            method_dst=(tx, ty),
            save_folder=step_dir,
            file_stem=stem,
            src_gt_box=anchor_size_ref if step == 1 else None,
            trg_gt_box=None,
            roi_size_ref_box=anchor_size_ref,
            device=device,
            hyper=hyper,
            experiment_type=label,
            experiment_detail=detail,
            with_tps_warp=with_tps_warp,
            tps_modes=tps_modes,
            semcorre_after_tps=semcorre_after_tps,
        )
        step_meta.append(
            {
                "step": step,
                "source_exam": panels[-1].exam_date,
                "target_exam": trg_date,
                "source_path": str(src_path),
                "target_path": str(trg_path),
                "source_kp_512": {"x": step_src_xy[0], "y": step_src_xy[1]},
                "target_kp_512": {"x": tx, "y": ty},
                "source_mode": source_mode,
                "target_has_own_roi": trg_own_xy is not None,
                "heatmap_score_at_pred": hm_at_pred,
                "heatmap_map_max": hm_max,
                "roi_iou_pred": roi_iou_pred,
                "roi_iou_heatmap_max_sum": roi_iou_heatmap,
                "roi_iou_heatmap_peak_sum": roi_iou_peak,
                "roi_gt_forward_target_kp_512": {"x": tx_roi, "y": ty_roi},
                "bidirectional_pair": bidirectional,
                "center_error": json.loads(
                    (step_dir / f"{stem}_center_error.json").read_text(encoding="utf-8")
                )
                if (step_dir / f"{stem}_center_error.json").is_file()
                else None,
            }
        )
        (chain_dir / f"{job.stem}_chain.json").write_text(
            json.dumps(
                {
                    "patient_id": job.patient_id,
                    "laterality": job.laterality,
                    "view": job.view,
                    "anchor_date": job.anchor_date,
                    "exam_timeline": dates,
                    "steps": step_meta,
                    "partial": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        trg_meta = lookup_exam_display_meta(
            index,
            job.patient_id,
            trg_date,
            job.laterality,
            job.view,
            trg_path,
        )
        panels.append(
            ChainPanelInfo(
                exam_date=trg_date,
                title=f"{trg_date}\n{trg_path.name}",
                display=trg_t,
                keypoint=(tx, ty),
                gt_box=trg_own_box,
                pred_box=trg_pred_box,
                heatmap_box=trg_heatmap_box,
                peak_box=trg_peak_box,
                white_box=trg_white_box,
                role=f"step {step} target",
                exam_year=trg_meta["exam_year"],
                exam_name=trg_meta["exam_name"],
                view_label=trg_meta["view_label"],
                exam_id=trg_meta["exam_id"],
                source_keypoint=step_src_xy,
                step_index=step,
                heatmap_score_at_kp=hm_at_pred,
                heatmap_map_max=hm_max,
            )
        )
        anchor_panels.append(
            ChainPanelInfo(
                exam_date=trg_date,
                title=f"{trg_date}\n{trg_path.name}",
                display=trg_t,
                keypoint=(tx_roi, ty_roi),
                gt_box=trg_own_box,
                pred_box=trg_pred_box_a,
                heatmap_box=trg_heatmap_box_a,
                peak_box=trg_peak_box_a,
                white_box=trg_white_box_a,
                role=f"ROI exam → {trg_date}",
                exam_year=trg_meta["exam_year"],
                exam_name=trg_meta["exam_name"],
                view_label=trg_meta["view_label"],
                exam_id=trg_meta["exam_id"],
                source_keypoint=roi_xy,
                step_index=step,
                heatmap_score_at_kp=hm_at_pred_a,
                heatmap_map_max=hm_max_a,
            )
        )
        src_path = trg_path
        src_t = trg_t
        if source_mode == "own_roi" and trg_own_xy is not None:
            src_xy_cur = trg_own_xy
            src_own_gt = trg_own_box
            print(
                f"  exp5 own_roi: {trg_date} has its own ROI; "
                "next hop uses that, not the transfer",
                flush=True,
            )
        else:
            src_xy_cur = (tx, ty)
            src_own_gt = trg_own_box if source_mode == "own_roi" else None
        detail = (
            f"patient {job.patient_id} | {job.laterality} {job.view} | "
            f"chain from ROI exam {seq_start_date} (timeline newest {timeline_newest}) → "
            f"{len(panels) - 1} older exam(s)"
        )
        overview_paths = write_exp5_overview_pngs(
            chain_dir=chain_dir,
            stem=job.stem,
            panels=panels,
            anchor_panels=anchor_panels,
            label=label,
            detail=detail,
            step_meta=step_meta,
            roi_size_ref_box=anchor_size_ref,
        )
        (chain_dir / f"{job.stem}_chain.json").write_text(
            json.dumps(
                {
                    "patient_id": job.patient_id,
                    "laterality": job.laterality,
                    "view": job.view,
                    "anchor_date": seq_start_date,
                    "roi_exam_date": roi_exam_date,
                    "timeline_newest_exam": timeline_newest,
                    "bridge_roi_to_newest": bridge_meta,
                    "exam_timeline": dates,
                    "steps": step_meta,
                    "partial": step < len(hop_dates),
                    **overview_paths,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not step_meta:
        detail = (
            f"patient {job.patient_id} | {job.laterality} {job.view} | "
            f"chain from ROI exam {seq_start_date} — no hops"
        )
        overview_paths = write_exp5_overview_pngs(
            chain_dir=chain_dir,
            stem=job.stem,
            panels=panels,
            anchor_panels=anchor_panels,
            label=label,
            detail=detail,
            step_meta=step_meta,
            roi_size_ref_box=anchor_size_ref,
        )
        (chain_dir / f"{job.stem}_chain.json").write_text(
            json.dumps(
                {
                    "patient_id": job.patient_id,
                    "laterality": job.laterality,
                    "view": job.view,
                    "anchor_date": chain_anchor_date,
                    "roi_exam_date": roi_exam_date,
                    "bridge_roi_to_newest": bridge_meta,
                    "exam_timeline": dates,
                    "steps": step_meta,
                    "partial": False,
                    **overview_paths,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def run_job(
    ldm,
    job: PairJob,
    run_root: Path,
    device: str,
    hyper: dict,
    *,
    with_tps_warp: bool = True,
    tps_modes: tuple[ControlMode, ...] = ("roi", "breast"),
    semcorre_after_tps: bool = True,
    pair_dir: Path | None = None,
) -> None:
    pair_dir = pair_dir or (run_root / job.experiment / job.stem)
    pair_dir.mkdir(parents=True, exist_ok=True)

    src_t = load_image_chw(job.src_path)
    trg_t = load_image_chw(job.trg_path)
    sx, sy = job.src_xy_512
    dataset = SinglePairDataset(src_t, trg_t, (sx, sy))
    mini_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))

    (
        est,
        _roi_iou_pred,
        _roi_iou_heatmap,
        _trg_gt,
        trg_pred_box,
        _trg_heatmap_box,
        _trg_peak_box,
        _roi_iou_peak,
        _trg_white_box,
        _hm_at,
        _hm_max,
    ) = run_correspondence(
        ldm,
        mini_batch,
        save_folder=pair_dir,
        file_stem=job.stem,
        source_path=str(job.src_path),
        target_path=str(job.trg_path),
        src_display=src_t,
        trg_display=trg_t,
        src_gt_box=job.src_gt_box_512,
        trg_gt_box=job.trg_gt_box_512,
        src_all_gt_boxes=job.src_all_gt_512,
        trg_all_gt_boxes=job.trg_all_gt_512,
        roi_size_ref_box=job.src_gt_box_512,
        device=device,
        experiment_type=job.experiment_label,
        experiment_detail=job.experiment_detail,
        **hyper,
    )
    run_tps_and_optional_semcorre_rerun(
        ldm,
        src_t=src_t,
        trg_t=trg_t,
        trg_path=job.trg_path,
        method_src=(float(sx), float(sy)),
        method_dst=(float(est[0].item()), float(est[1].item())),
        save_folder=pair_dir,
        file_stem=job.stem,
        src_gt_box=job.src_gt_box_512,
        trg_gt_box=job.trg_gt_box_512,
        roi_size_ref_box=job.src_gt_box_512,
        device=device,
        hyper=hyper,
        experiment_type=job.experiment_label,
        experiment_detail=job.experiment_detail,
        with_tps_warp=with_tps_warp,
        tps_modes=tps_modes,
        semcorre_after_tps=semcorre_after_tps,
    )
    roundtrip = run_roundtrip_if_needed(
        ldm,
        src_gt_box=job.src_gt_box_512,
        trg_gt_box=job.trg_gt_box_512,
        trg_t=trg_t,
        src_t=src_t,
        forward_target_kp=(float(est[0].item()), float(est[1].item())),
        forward_src_kp=(float(sx), float(sy)),
        original_src_path=job.src_path,
        original_trg_path=job.trg_path,
        save_folder=pair_dir,
        file_stem=job.stem,
        device=device,
        hyper=hyper,
        experiment_type=job.experiment_label,
        experiment_detail=job.experiment_detail,
        forward_trg_white_box=_trg_white_box,
        forward_trg_pred_box=trg_pred_box,
        src_all_gt_boxes=job.src_all_gt_512,
        trg_all_gt_boxes=job.trg_all_gt_512,
    )
    meta = asdict(job)
    meta["src_path"] = str(job.src_path)
    meta["trg_path"] = str(job.trg_path)
    meta["output_dir"] = str(pair_dir)
    meta["roundtrip_back"] = roundtrip
    err_path = pair_dir / f"{job.stem}_center_error.json"
    if err_path.is_file():
        meta["center_error"] = json.loads(err_path.read_text(encoding="utf-8"))
    (pair_dir / f"{job.stem}_pair.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


EXPERIMENT_HELP = """
Experiment ids (use in --experiments):
  exp1  MLO ↔ CC (both directions), same patient / exam / laterality
  exp2  L → R and R → L, same patient / exam / view
  exp3  other patients → anchor; same lat+view as anchor (--anchor-patient)
  exp4  ROI exam → prior exam, same lat / view (no target GT)
  exp5  chain backward from newest exam; ROI GT from roi_coords (bridge if ROI exam is older)
        Filter chains: --exp5-views PATIENT:LAT:VIEW,...  e.g. 62877247:L:CC
        Default: full timeline. Quick test only: --exp5-max-steps 1
  exp6  Same pairs as exp1 (CC↔MLO); use --layers 2 3 4 5 6 (combined mid stack)
  exp7  Same pairs as exp1; layer ablation — run once with --layers 2..10 and once with 7 8 9 10 (baseline)
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch mammogram correspondence experiments.",
        epilog=EXPERIMENT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pack-dir",
        type=str,
        default="../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5",
        help="Path to roi_overlays_cancer5 pack",
    )
    p.add_argument("--out-dir", type=str, default="outputs/batch_experiments")
    p.add_argument(
        "--run-tag",
        type=str,
        default="",
        help=(
            "Prefix for subfolder under --out-dir (default: UTC YYYY-MM-DD_HHMMSS only). "
            "A timestamp is appended automatically, e.g. my_exp5 → my_exp5_2025-09-25_031500"
        ),
    )
    p.add_argument(
        "--run-tag-exact",
        action="store_true",
        help="Use --run-tag exactly with no date suffix (overwrites that folder on rerun)",
    )
    p.add_argument(
        "--experiments",
        type=str,
        default="exp1,exp2,exp3,exp4",
        help="Comma list, e.g. exp1 or exp1,exp2 (see epilog below)",
    )
    p.add_argument(
        "--anchor-patient",
        type=str,
        default="all",
        help="Exp3: target patient id, or 'all' for every patient's ROI rows as anchors",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="Max pairs total (0 = all)")
    p.add_argument(
        "--pair-stems",
        type=str,
        default="",
        help="Comma-separated pair stems to run (exact job.stem). Also pair_stems in --set-json.",
    )
    p.add_argument(
        "--exp12-patient",
        type=str,
        default="",
        help="If set, exp1 and exp2 only for this patient id (overridden by --exp1-patient / --exp2-patient)",
    )
    p.add_argument(
        "--exp1-patient",
        type=str,
        default="",
        help="If set, exp1 only for this patient id",
    )
    p.add_argument(
        "--exam",
        action="append",
        dest="exams",
        default=None,
        help="Restrict exp1/exp2 to this patient:YYYY-MM-DD (repeatable)",
    )
    p.add_argument(
        "--set-json",
        type=str,
        default="",
        help=(
            "Review set JSON from scripts/build_exp_sets_from_review.py "
            "(data/exp_sets/exp1_views.json). Restricts exp1/exp2 to those exams "
            "and exp5 to those patient/lat/view chains."
        ),
    )
    p.add_argument(
        "--exp2-patient",
        type=str,
        default="",
        help="If set, exp2 only for this patient id",
    )
    p.add_argument(
        "--exp3-max-sources",
        type=int,
        default=0,
        help="Exp3: max source pairs (0 = all non-anchor sources)",
    )
    p.add_argument(
        "--exp3-source-patient",
        type=str,
        default="",
        help="Exp3: only use this patient id as source (empty = any other patient)",
    )
    p.add_argument(
        "--exp5-max-patients",
        type=int,
        default=0,
        help="Exp5: max distinct patients (0 = all); keeps all chains for those patients",
    )
    p.add_argument(
        "--exp5-views",
        type=str,
        default="",
        help=(
            "Exp5: run only these (patient_id, laterality, view) chains, comma-separated. "
            "Format: PATIENT:L:VIEW or PATIENT:R:VIEW (e.g. 62877247:L:CC,45209155:R:MLO). "
            "Each needs a matching row in roi_coords.csv for that lat+view. Empty = all chains."
        ),
    )
    p.add_argument(
        "--exp5-max-steps",
        type=int,
        default=0,
        help=(
            "Exp5: max backward hops per chain (0 = full timeline). "
            "Use 1 for a single pair: ROI anchor exam → immediately prior exam only."
        ),
    )
    p.add_argument(
        "--exp5-source-mode",
        type=str,
        default="transfer",
        choices=("transfer", "own_roi"),
        help=(
            "Exp5 next-hop source. transfer: always last-year predicted point "
            "(still draws any exam's own ROI). own_roi: if that exam has an ROI, "
            "use it as the next source instead of the transfer."
        ),
    )
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
        help=(
            "Token-optimization GT. gaussian: circle around the ROI center (default). "
            "roi_box: binary mask, 1 inside the source ROI and 0 outside (no Gaussian)."
        ),
    )
    p.add_argument(
        "--gt-compare",
        action="store_true",
        help="Run each pair twice (Gaussian GT and ROI-box GT) and save side-by-side plots.",
    )
    p.add_argument("--crop_percent", type=float, default=93.16549294381423)
    p.add_argument("--flip_prob", type=float, default=0.0)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[7, 8, 9, 10],
        help="DHPF attention layers (default 7–10, production setting)",
    )
    p.add_argument("--model_type", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--upsample_res", type=int, default=512)
    p.add_argument(
        "--with-tps-warp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each unwarped SemCorre pair, save TPS before/after PNGs (default: on)",
    )
    p.add_argument(
        "--tps-modes",
        type=str,
        default="roi,breast",
        help="TPS control sets: roi (ROI box corners+center), breast (10 uniform tissue points). Comma list.",
    )
    p.add_argument(
        "--semcorre-after-tps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After TPS, run SemCorre again on warped source vs same target "
            "(one run per --tps-modes entry; default: off — much slower if on)"
        ),
    )
    p.add_argument(
        "--rebuild-exp5-overviews",
        type=str,
        default="",
        help=(
            "Rebuild exp5 chain_overview PNGs from existing *_chain.json under this folder "
            "(no Stable Diffusion). Does not start a new experiment run."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip pair jobs that already have {stem}_pair.json, and exp5 chains whose "
            "*_chain.json has partial=false. Use with --run-tag-exact to resume a run folder."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    pack_root = Path(args.pack_dir).expanduser().resolve()
    if args.rebuild_exp5_overviews:
        search = Path(args.rebuild_exp5_overviews).expanduser().resolve()
        jsons = sorted(search.rglob("*_chain.json"))
        if not jsons:
            print(f"No *_chain.json under {search}")
            return
        print(f"Rebuilding exp5 overviews from {len(jsons)} chain.json file(s) under {search}")
        for jp in jsons:
            print(f"  {jp}", flush=True)
            try:
                paths = rebuild_exp5_overviews_from_chain_json(jp, pack_root)
                for k, v in paths.items():
                    print(f"    {k}: {v}")
            except Exception as exc:
                print(f"    FAILED: {exc}")
        return
    out_base = Path(args.out_dir).expanduser().resolve()
    run_tag = resolve_run_tag(args.run_tag, exact=args.run_tag_exact)
    run_root = out_base / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Run tag: {run_tag}", flush=True)

    records = load_roi_table(pack_root)
    index = load_index(pack_root)
    print(f"Loaded {len(records)} ROI records from pack {pack_root}")

    want_list = [x.strip() for x in args.experiments.split(",") if x.strip()]
    want = set(want_list)
    jobs: list[PairJob] = []
    exp12_pid = (args.exp12_patient or "").strip()
    exp1_pid = (args.exp1_patient or exp12_pid).strip()
    exp2_pid = (args.exp2_patient or exp12_pid).strip()
    exp1_set = {exp1_pid} if exp1_pid else set()
    exp2_set = {exp2_pid} if exp2_pid else set()

    exam_specs = parse_exam_specs(args.exams)
    set_info = load_set_json(args.set_json) if args.set_json else None
    if set_info:
        print(
            f"Set JSON {args.set_json}: experiment={set_info['experiment']} "
            f"patients={len(set_info['patient_ids'])} exams={len(set_info['exam_specs'])}",
            flush=True,
        )
        if set_info["exam_specs"]:
            exam_specs = exam_specs | set_info["exam_specs"] if exam_specs else set_info["exam_specs"]
        if set_info["experiment"] in ("exp1", "exp6", "exp7") and set_info["patient_ids"]:
            exp1_set = exp1_set | set_info["patient_ids"] if exp1_set else set_info["patient_ids"]
        if set_info["experiment"] == "exp2" and set_info["patient_ids"]:
            exp2_set = exp2_set | set_info["patient_ids"] if exp2_set else set_info["patient_ids"]
    if "exp1" in want:
        jobs.extend(_exp1_pairs_for_set(records, pack_root, index, exp1_set, exam_specs))
    if "exp6" in want:
        jobs.extend(
            _relabel_exp6_job(j)
            for j in _exp1_pairs_for_set(records, pack_root, index, exp1_set, exam_specs)
        )
    if "exp7" in want:
        jobs.extend(
            _relabel_exp7_job(j)
            for j in _exp1_pairs_for_set(records, pack_root, index, exp1_set, exam_specs)
        )
    if "exp2" in want:
        if (
            set_info
            and set_info["experiment"] == "exp2"
            and set_info.get("review_patients")
        ):
            j2 = build_exp2_from_set(
                records, pack_root, index, set_info["review_patients"]
            )
        else:
            j2 = build_exp2(records, pack_root, index)
            j2 = filter_pair_jobs_for_patients(j2, exp2_set) if exp2_set else j2
            j2 = filter_pair_jobs_for_exams(j2, exam_specs)
        jobs.extend(j2)
    prim = primary_records(records)
    if "exp3" in want:
        if set_info and set_info.get("exp3_groups"):
            jobs.extend(build_exp3_from_set(prim, pack_root, set_info["exp3_groups"]))
        else:
            jobs.extend(
                build_exp3(
                    prim,
                    pack_root,
                    args.anchor_patient,
                    max_sources=args.exp3_max_sources,
                    source_patient=args.exp3_source_patient,
                )
            )
    if "exp4" in want:
        jobs.extend(build_exp4(prim, pack_root, index))

    chain_jobs: list[ChainJob] = []
    if "exp5" in want:
        exp5_specs = parse_exp5_view_specs(args.exp5_views)
        if set_info and set_info["experiment"] == "exp5" and set_info["exp5_views"]:
            exp5_specs = exp5_specs | set_info["exp5_views"] if exp5_specs else set_info["exp5_views"]
        all_exp5 = build_exp5(prim, pack_root, index)
        chain_jobs = filter_exp5_chains(all_exp5, exp5_specs)
        chain_jobs = limit_exp5_chains(chain_jobs, args.exp5_max_patients)
        if exp5_specs and not chain_jobs:
            print("WARNING: --exp5-views matched no chains. Check roi_coords.csv for these specs:", flush=True)
            for pid, lat, view in sorted(exp5_specs):
                print(f"  requested: patient {pid}  {lat}-{view}", flush=True)
            print("  available exp5 chains from pack:", flush=True)
            for c in all_exp5:
                print(f"    {c.patient_id}  {c.laterality} {c.view}  anchor {c.anchor_date}", flush=True)

    pair_stems: set[str] = set()
    if args.pair_stems.strip():
        pair_stems.update(s.strip() for s in args.pair_stems.split(",") if s.strip())
    if set_info and set_info.get("pair_stems"):
        pair_stems.update(set_info["pair_stems"])
    if pair_stems:
        n_before = len(jobs)
        jobs = filter_pair_jobs_by_stems(jobs, pair_stems)
        print(
            f"Pair stem filter: {len(jobs)} job(s) kept ({n_before} before filter)",
            flush=True,
        )
        for j in jobs:
            print(f"  • {j.stem}  [{j.experiment}]", flush=True)

    if args.limit > 0:
        jobs = jobs[: args.limit]
        chain_jobs = chain_jobs[: args.limit]

    print(f"Planned {len(jobs)} pairs across experiments (order {want_list}): {want}")
    if chain_jobs:
        print(f"Planned {len(chain_jobs)} sequential chains (exp5):")
        for c in chain_jobs:
            print(f"  • patient {c.patient_id}  {c.laterality} {c.view}  from {c.anchor_date}")
    print(f"Run output root: {run_root}")
    if not jobs and not chain_jobs:
        return

    tps_modes = parse_tps_modes(args.tps_modes)

    (run_root / "run_info.json").write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiments": sorted(want),
                "n_pairs": len(jobs),
                "n_chains": len(chain_jobs),
                "pack_dir": str(pack_root),
                "exp12_patient": exp12_pid or None,
                "exp1_patient": exp1_pid or None,
                "exp2_patient": exp2_pid or None,
                "exp3_max_sources": args.exp3_max_sources,
                "exp3_source_patient": args.exp3_source_patient or None,
                "exp5_max_patients": args.exp5_max_patients,
                "exp5_views": args.exp5_views or None,
                "exp5_max_steps": args.exp5_max_steps,
                "num_opt_iterations": args.num_opt_iterations,
                "num_iterations": args.num_iterations,
                "layers": list(args.layers),
                "set_json": args.set_json or None,
                "gt_mode": args.gt_mode,
                "gt_compare": args.gt_compare,
                "with_tps_warp": args.with_tps_warp,
                "tps_modes": list(tps_modes),
                "semcorre_after_tps": args.semcorre_after_tps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading Stable Diffusion on {device}...")
    ldm = load_ldm(device, args.model_type)

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
        gt_mode=args.gt_mode,
    )

    def _run_pair(i: int, n: int, job: PairJob) -> bool:
        if args.skip_existing and pair_job_already_done(run_root, job):
            print(f"\n[{i}/{n}] SKIP (done) {job.experiment} | {job.stem}", flush=True)
            return True
        print(f"\n[{i}/{n}] {job.experiment} | {job.stem}", flush=True)
        try:
            if args.gt_compare:
                base = run_root / job.experiment / job.stem
                gdir = base / "gt_gaussian"
                rdir = base / "gt_roi_box"
                print("  GT compare: Gaussian…", flush=True)
                run_job(
                    ldm, job, run_root, device, {**hyper, "gt_mode": "gaussian"},
                    with_tps_warp=args.with_tps_warp, tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps, pair_dir=gdir,
                )
                print("  GT compare: ROI-box…", flush=True)
                run_job(
                    ldm, job, run_root, device, {**hyper, "gt_mode": "roi_box"},
                    with_tps_warp=args.with_tps_warp, tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps, pair_dir=rdir,
                )
                compose_gt_compare_sheet(gdir, rdir, base)
            else:
                run_job(
                    ldm,
                    job,
                    run_root,
                    device,
                    hyper,
                    with_tps_warp=args.with_tps_warp,
                    tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps,
                )
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
        return False

    def _run_chain(i: int, n: int, cjob: ChainJob) -> bool:
        if args.skip_existing and chain_job_already_done(run_root, cjob):
            print(f"\n[chain {i}/{n}] SKIP (done) exp5 | {cjob.stem}", flush=True)
            return True
        print(f"\n[chain {i}/{n}] exp5 | {cjob.stem}", flush=True)
        try:
            if args.gt_compare:
                base = run_root / "exp5_sequential_prior" / cjob.stem
                gdir = base / "gt_gaussian"
                rdir = base / "gt_roi_box"
                print("  GT compare: Gaussian…", flush=True)
                run_chain_job(
                    ldm, cjob, run_root, pack_root, index, device,
                    {**hyper, "gt_mode": "gaussian"},
                    with_tps_warp=args.with_tps_warp, tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps,
                    exp5_max_steps=args.exp5_max_steps, chain_dir=gdir,
                    records=records, source_mode=args.exp5_source_mode,
                )
                print("  GT compare: ROI-box…", flush=True)
                run_chain_job(
                    ldm, cjob, run_root, pack_root, index, device,
                    {**hyper, "gt_mode": "roi_box"},
                    with_tps_warp=args.with_tps_warp, tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps,
                    exp5_max_steps=args.exp5_max_steps, chain_dir=rdir,
                    records=records, source_mode=args.exp5_source_mode,
                )
                for step_g in sorted(gdir.glob("step_*")):
                    step_r = rdir / step_g.name
                    if step_r.is_dir():
                        compose_gt_compare_sheet(step_g, step_r, base / step_g.name)
                compose_gt_compare_sheet(gdir, rdir, base)
            else:
                run_chain_job(
                    ldm,
                    cjob,
                    run_root,
                    pack_root,
                    index,
                    device,
                    hyper,
                    with_tps_warp=args.with_tps_warp,
                    tps_modes=tps_modes,
                    semcorre_after_tps=args.semcorre_after_tps,
                    exp5_max_steps=args.exp5_max_steps,
                    records=records,
                    source_mode=args.exp5_source_mode,
                )
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
        return False

    skipped_pairs = 0
    skipped_chains = 0
    pair_i = 0
    chain_i = 0
    for token in want_list:
        if token == "exp5":
            for cjob in chain_jobs:
                chain_i += 1
                if _run_chain(chain_i, len(chain_jobs), cjob):
                    skipped_chains += 1
            continue
        matching = [j for j in jobs if j.experiment.startswith(token)]
        for job in matching:
            pair_i += 1
            if _run_pair(pair_i, len(jobs), job):
                skipped_pairs += 1
    if skipped_pairs:
        print(f"\nSkipped {skipped_pairs} pair job(s) (--skip-existing).", flush=True)
    if skipped_chains:
        print(f"\nSkipped {skipped_chains} exp5 chain(s) (--skip-existing).", flush=True)

    write_center_error_summary(run_root)
    print(f"\nDone. All results under:\n  {run_root}/<experiment>/<pair-stem>/", flush=True)


if __name__ == "__main__":
    main()
