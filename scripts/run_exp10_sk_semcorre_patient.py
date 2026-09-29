"""Experiment 10 orchestrator (runs from SemCorre repo; imports StableKeypointsPlus).

Set ``SKP`` / ``SKP_ROOT`` to your StableKeypointsPlus clone (Vista default below).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SEMCRE_REPO = Path(__file__).resolve().parents[1]


def _skp_root() -> Path:
    for key in ("SKP", "SKP_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                return p.resolve()
    default = Path("/home1/11364/paliniya/projects/StableKeypointsPlus")
    if default.is_dir():
        return default
    raise SystemExit(
        "StableKeypointsPlus not found. Export SKP=/path/to/StableKeypointsPlus "
        "(needs run_embed_patient_model.py, datasets/, sk_viz.py)."
    )


REPO = _skp_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.embed_image import MODEL_LETTERBOX_SIZE  # noqa: E402
from run_graphmatch_patient_smoke import collect_patient_model  # noqa: E402


def _resolve_exp5_set_path(set_json: Path) -> Path:
    set_json = set_json.resolve()
    data = json.loads(set_json.read_text(encoding="utf-8"))
    ref = data.get("same_as_exp5_set") or data.get("exp5_set_json")
    if ref:
        cand = set_json.parent / ref
        if cand.is_file():
            return cand
        cand2 = set_json.parent.parent / "exp_sets" / Path(ref).name
        if cand2.is_file():
            return cand2
    return set_json


def load_exp5_patient_buckets(set_path: Path) -> list[dict]:
    data = json.loads(set_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for rec in data.get("patients") or []:
        pid = str(rec.get("id") or "")
        if not pid:
            continue
        buckets: list[str] = []
        for ch in rec.get("chains") or []:
            lat = str(ch.get("laterality") or "")[:1].upper()
            view = str(ch.get("view") or "").upper()
            if lat and view:
                buckets.append(f"{lat}_{view}")
        if buckets:
            out.append({"patient_id": pid, "buckets": buckets})
    if not out and data.get("patient_ids"):
        for pid in data["patient_ids"]:
            out.append({"patient_id": str(pid), "buckets": ["L_CC", "R_CC", "L_MLO", "R_MLO"]})
    return out


def _run_sk_arm(
    patient_id: str,
    exam_order: str,
    args: argparse.Namespace,
    arm_dir: Path,
    token: str,
) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "run_embed_patient_model.py"),
        "--patient_id",
        patient_id,
        "--dataset_loc",
        str(args.dataset_loc),
        "--output_root",
        str(arm_dir),
        "--num_train_exams",
        "1",
        "--exam_order",
        exam_order,
        "--top_k",
        str(args.top_k),
        "--views",
        "CC",
        "MLO",
        "--my_token",
        token,
    ]
    if args.clinical_csv:
        cmd.extend(["--clinical_csv", str(args.clinical_csv)])
    if args.skip_build:
        cmd.append("--skip_build")
    if args.visualize:
        cmd.append("--visualize")
    if args.mammoclip_crop:
        cmd.append("--mammoclip_crop")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _draw_temporal_strip(items: list[dict], out_path: Path, *, top_k: int, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from sk_viz import exam_panel

    if not items:
        return
    n_k = min(top_k, min(it["xy"].shape[0] for it in items))
    panels = []
    for it in items:
        panels.append(
            exam_panel(
                it["path"],
                it["xy"][:n_k],
                conf=it.get("conf"),
                faces=None,
                title=it.get("label"),
                viz_size=MODEL_LETTERBOX_SIZE,
                chrome="inside",
                kpt_ids=list(range(n_k)),
            )
        )
    w, h = panels[0].size
    gap = 10
    banner = 28
    strip = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, h + banner), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    for i, p in enumerate(panels):
        strip.paste(p, (i * (w + gap), banner))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)
    print("wrote", out_path, flush=True)


def _export_arm_json(
    patient_id: str,
    arm_name: str,
    exam_order: str,
    arm_dir: Path,
    dataset_loc: Path,
    top_k: int,
    buckets: list[str],
) -> dict:
    kpt_root = arm_dir / "results" / f"patient_{patient_id}"
    png_root = arm_dir / f"patient_{patient_id}"
    chain_direction = "forward" if exam_order == "first" else "backward"
    train_exam_date = None
    views_out: dict[str, list] = {}
    for bucket in buckets:
        items = collect_patient_model(
            kpt_root,
            png_root,
            patient_id,
            bucket,
            top_k=top_k,
            img_size=MODEL_LETTERBOX_SIZE,
            include_train=True,
            dataset_loc=dataset_loc,
        )
        if len(items) < 2:
            print(f"  [exp10] skip {patient_id} {bucket}: need >=2 exams, got {len(items)}", flush=True)
            continue
        n_k = min(top_k, min(it["xy"].shape[0] for it in items))
        rows = []
        for it in items:
            date = str(it["meta"].get("exam_date") or it["label"])[:10]
            rows.append(
                {
                    "exam_date": date,
                    "png_path": str(it["path"]),
                    "xy_512": it["xy"][:n_k].tolist(),
                    "is_train_source": "/train/" in str(it["path"]).replace("\\", "/"),
                }
            )
        rows.sort(key=lambda r: r["exam_date"])
        train_exam_date = rows[0]["exam_date"] if exam_order == "first" else rows[-1]["exam_date"]
        views_out[bucket] = rows
    return {
        "arm": arm_name,
        "exam_order": exam_order,
        "train_exam_date": train_exam_date,
        "chain_direction": chain_direction,
        "top_k": top_k,
        "buckets": views_out,
    }


def process_one_patient(
    pid: str,
    buckets: list[str],
    args: argparse.Namespace,
    token: str,
    patient_out: Path,
) -> dict:
    patient_out.mkdir(parents=True, exist_ok=True)
    arms = (
        ("train_first", "first", patient_out / "sk_train_first"),
        ("train_last", "last", patient_out / "sk_train_last"),
    )
    if not args.skip_sk_train:
        for _name, order, arm_dir in arms:
            _run_sk_arm(pid, order, args, arm_dir, token)

    patient_export: dict = {"patient_id": pid, "exp5_buckets": buckets, "arms": {}}
    fig_dir = patient_out / "figures" / "temporal_kpts_no_triangles"
    for arm_name, order, arm_dir in arms:
        patient_export["arms"][arm_name] = _export_arm_json(
            pid, arm_name, order, arm_dir, args.dataset_loc, args.top_k, buckets
        )
        kpt_root = arm_dir / "results" / f"patient_{pid}"
        png_root = arm_dir / f"patient_{pid}"
        for bucket in buckets:
            items = collect_patient_model(
                kpt_root,
                png_root,
                pid,
                bucket,
                top_k=args.top_k,
                img_size=MODEL_LETTERBOX_SIZE,
                include_train=True,
                dataset_loc=args.dataset_loc,
            )
            if len(items) < 2:
                continue
            _draw_temporal_strip(
                items,
                fig_dir / f"{arm_name}_{bucket}_temporal.png",
                top_k=args.top_k,
                title=f"Exp10 {pid} {arm_name} {bucket} (K={args.top_k}, no triangles)",
            )
    return patient_export


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-json", type=Path, default=None)
    p.add_argument("--patient-id", action="append", dest="patient_ids")
    p.add_argument("--dataset-loc", type=Path, required=True)
    p.add_argument("--pack-dir", type=Path)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--token-file", type=Path, default=None)
    p.add_argument("--clinical-csv", type=Path, default=None)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--mammoclip-crop", action="store_true")
    p.add_argument("--sk-only", action="store_true")
    p.add_argument("--skip-sk-train", action="store_true")
    p.add_argument("--semcorre-repo", type=Path, default=SEMCRE_REPO)
    args = p.parse_args()

    print(f"Exp10 SKP root: {REPO}", flush=True)
    semcorre_repo = args.semcorre_repo.resolve()
    default_set = semcorre_repo / "data" / "exp_sets" / "exp5_temporal.json"
    set_json = args.set_json or default_set
    if not set_json.is_file():
        raise SystemExit(f"Missing set JSON: {set_json}")
    exp5_path = _resolve_exp5_set_path(set_json)
    print(f"Exp10 using Exp5 set: {exp5_path}", flush=True)

    token_file = args.token_file or (REPO / "token.txt")
    args.token_file = token_file

    cohort = load_exp5_patient_buckets(exp5_path)
    want = set(args.patient_ids or [])
    if want:
        cohort = [c for c in cohort if c["patient_id"] in want]
    if not cohort:
        raise SystemExit("No patients to run")

    if not args.token_file.is_file() and not args.skip_sk_train:
        raise SystemExit(f"Missing HF token: {args.token_file}")
    token = args.token_file.read_text(encoding="utf-8").strip() if args.token_file.is_file() else ""

    args.out_root.mkdir(parents=True, exist_ok=True)
    export = {
        "experiment": "exp10",
        "exp5_set_json": str(exp5_path),
        "skp_root": str(REPO),
        "dataset_loc": str(args.dataset_loc),
        "pack_dir": str(args.pack_dir) if args.pack_dir else None,
        "top_k": args.top_k,
        "patients": {},
    }

    for spec in cohort:
        pid = spec["patient_id"]
        print(f"\n=== Exp10 patient {pid} buckets={spec['buckets']} ===", flush=True)
        export["patients"][pid] = process_one_patient(
            pid, spec["buckets"], args, token, args.out_root / f"patient_{pid}"
        )

    export_path = args.out_root / "exp10_sk_keypoints.json"
    export_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print("Wrote", export_path, flush=True)

    if args.sk_only or not args.pack_dir:
        return

    cmd = [
        sys.executable,
        str(semcorre_repo / "scripts" / "run_exp10_sk_kpt_chains.py"),
        "--export-json",
        str(export_path),
        "--pack-dir",
        str(args.pack_dir),
        "--out-dir",
        str(args.out_root / "semcorre_chains"),
        "--device",
        "cuda:0",
    ]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(semcorre_repo))


if __name__ == "__main__":
    main()
