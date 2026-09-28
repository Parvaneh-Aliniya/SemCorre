"""Build exp1/2/3/5 set JSON from Experiments review folders."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

PNG = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<lat>[LR])_(?P<view>CC|MLO)_(?P<stem>.+)\.png$",
    re.I,
)
PID = re.compile(r"patient_(\d+)")

ROOT = Path(__file__).resolve().parents[1]
REVIEW = (
    ROOT
    / "outputs"
    / "batch_experiments"
    / "Experiments"
)
OUT = ROOT / "data" / "exp_sets"


def _pid(name: str) -> str | None:
    m = PID.search(name)
    return m.group(1) if m else None


def _images(folder: Path) -> list[dict]:
    rows = []
    for p in sorted(folder.glob("*.png")):
        m = PNG.match(p.name)
        if not m:
            continue
        rows.append(
            {
                "date": m.group("date"),
                "laterality": m.group("lat").upper(),
                "view": m.group("view").upper(),
                "stem": m.group("stem"),
                "file": p.name,
            }
        )
    return rows


def _group_exams(images: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for im in images:
        by[(im["date"], im["laterality"])].append(im)
    exams = []
    for (day, lat), ims in sorted(by.items()):
        exams.append(
            {
                "date": day,
                "laterality": lat,
                "views": sorted({i["view"] for i in ims}),
                "images": ims,
            }
        )
    return exams


def parse_folder_view_tags(folder: str) -> list[tuple[str, str]]:
    """Read L/R + CC/MLO from a review folder name. Empty = no view tag."""
    n = folder.lower()
    if "all_four" in n or "allfour" in n:
        return [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]
    compact = re.sub(r"[^a-z0-9]", "", n)
    found: list[tuple[str, str]] = []
    if re.search(r"(^|_)l_?cc(_|$)", n) or "lcc" in compact:
        found.append(("L", "CC"))
    if re.search(r"(^|_)l_?mlo(_|$)", n) or "lmlo" in compact:
        found.append(("L", "MLO"))
    if re.search(r"(^|_)r_?cc(_|$)", n) or "rcc" in compact:
        found.append(("R", "CC"))
    if re.search(r"(^|_)r_?mlo(_|$)", n) or "rmlo" in compact:
        found.append(("R", "MLO"))
    if found:
        out: list[tuple[str, str]] = []
        for item in found:
            if item not in out:
                out.append(item)
        return out
    lats: list[str] = []
    if "left" in n:
        lats.append("L")
    if "right" in n:
        lats.append("R")
    views: list[str] = []
    if re.search(r"(^|_)cc(_|$)|cc_and|and_cc", n) or re.search(r"cc", n):
        views.append("CC")
    if "mlo" in n:
        views.append("MLO")
    return [(la, v) for la in lats for v in views]


def collect_exp5_tagged() -> tuple[list[dict], list[dict]]:
    """Only themporal folders tagged with left/right + CC/MLO."""
    folder = REVIEW / "themporal"
    patients = []
    chains = []
    for child in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not child.is_dir():
            continue
        pid = _pid(child.name)
        if not pid:
            continue
        tags = parse_folder_view_tags(child.name)
        if not tags:
            print(f"  skip untagged temporal folder {child.name}")
            continue
        images = _images(child)
        rec = {
            "id": pid,
            "folder": child.name,
            "chains": [{"laterality": la, "view": v} for la, v in tags],
            "exams": _group_exams(images),
            "note": "Pack ALL exams of each tagged lat+view (ROI and non-ROI).",
        }
        patients.append(rec)
        for la, v in tags:
            chains.append(
                {
                    "patient_id": pid,
                    "laterality": la,
                    "view": v,
                    "folder": child.name,
                }
            )
    return patients, chains


def collect_exp3_groups() -> list[dict]:
    """One group per small/large folder: *_target is the anchor, others are sources."""
    groups = []
    root = REVIEW / "cross_patient"
    for bucket in ("small", "large"):
        folder = root / bucket
        if not folder.is_dir():
            continue
        people = []
        target = None
        for child in sorted(folder.iterdir()):
            if not child.is_dir():
                continue
            pid = _pid(child.name)
            if not pid:
                continue
            images = _images(child)
            if not images:
                continue
            rec = {
                "id": pid,
                "folder": child.name,
                "is_target": "_target" in child.name.lower(),
                "exams": _group_exams(images),
                "images": images,
            }
            people.append(rec)
            if rec["is_target"]:
                target = rec
        if target is None:
            print(f"WARNING: no *_target folder in {folder}")
            continue
        sources = [p for p in people if p["id"] != target["id"]]
        pairs = []
        trg_by_view: dict[str, list[dict]] = defaultdict(list)
        for im in target["images"]:
            trg_by_view[im["view"]].append(im)
        for src in sources:
            for sim in src["images"]:
                for tim in trg_by_view.get(sim["view"], []):
                    pairs.append(
                        {
                            "source_id": src["id"],
                            "source_date": sim["date"],
                            "source_laterality": sim["laterality"],
                            "source_view": sim["view"],
                            "source_stem": sim["stem"],
                            "target_id": target["id"],
                            "target_date": tim["date"],
                            "target_laterality": tim["laterality"],
                            "target_view": tim["view"],
                            "target_stem": tim["stem"],
                        }
                    )
        groups.append(
            {
                "bucket": bucket,
                "target": {k: target[k] for k in ("id", "folder", "exams", "images")},
                "sources": [
                    {k: s[k] for k in ("id", "folder", "exams", "images")} for s in sources
                ],
                "pairs": pairs,
            }
        )
    return groups


def collect(folder: Path, extra: dict | None = None) -> list[dict]:
    patients = []
    if not folder.is_dir():
        return patients
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        pid = _pid(child.name)
        if not pid:
            continue
        images = _images(child)
        if not images:
            continue
        rec = {
            "id": pid,
            "folder": child.name,
            "exams": _group_exams(images),
        }
        if extra:
            rec.update(extra)
        patients.append(rec)
    return patients


def write_set(name: str, experiment: str, source: str, patients: list[dict], note: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    exams = []
    for p in patients:
        for ex in p["exams"]:
            exams.append(f"{p['id']}:{ex['date']}")
    payload = {
        "name": name,
        "experiment": experiment,
        "source": source,
        "note": note,
        "n_patients": len(patients),
        "exam_specs": sorted(set(exams)),
        "patient_ids": [p["id"] for p in patients],
        "patients": patients,
    }
    dest = OUT / f"{name}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {dest}  patients={len(patients)} exams={len(payload['exam_specs'])}")
    return dest


def main() -> None:
    views = REVIEW / "views"
    write_set(
        "exp1_views",
        "exp1",
        "Experiments/views",
        collect(views),
        "Exp1 test set: CC↔MLO on the exam images you put in each patient folder.",
    )
    write_set(
        "exp2_lateral",
        "exp2",
        "Experiments/lateral",
        collect(REVIEW / "lateral"),
        "Exp2 review set from Experiments/lateral. Images may be different dates (not always same-exam L↔R).",
    )
    groups = collect_exp3_groups()
    exp3_patients = []
    for g in groups:
        t = dict(g["target"])
        t["is_target"] = True
        t["bucket"] = g["bucket"]
        exp3_patients.append(t)
        for s in g["sources"]:
            rec = dict(s)
            rec["is_target"] = False
            rec["bucket"] = g["bucket"]
            exp3_patients.append(rec)
    dest = write_set(
        "exp3_cross_patient",
        "exp3",
        "Experiments/cross_patient",
        exp3_patients,
        "Exp3: each small/large folder has one *_target. Every other patient in that "
        "folder is a source, paired to the target on the same view (CC→CC, MLO→MLO).",
    )
    payload = json.loads(dest.read_text(encoding="utf-8"))
    payload["groups"] = groups
    payload["n_pairs"] = sum(len(g["pairs"]) for g in groups)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  exp3 groups={len(groups)} pairs={payload['n_pairs']}")
    exp5_patients, exp5_chains = collect_exp5_tagged()
    dest5 = write_set(
        "exp5_temporal",
        "exp5",
        "Experiments/themporal",
        exp5_patients,
        "Only folders tagged left/right + CC/MLO. Each tag is one chain. "
        "Timeline = every exam of that lat+view (with or without ROI). "
        "Two result modes: transfer (always last-year prediction) and own_roi "
        "(use that exam's ROI as source when it has one). Both still draw any native ROI.",
    )
    payload = json.loads(dest5.read_text(encoding="utf-8"))
    payload["chains"] = exp5_chains
    dest5.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  exp5 tagged chains={len(exp5_chains)}")


if __name__ == "__main__":
    main()
