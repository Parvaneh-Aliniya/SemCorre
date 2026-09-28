r"""Mammography ROI centers exported from StableKeypointsPlus.

Expects ``data/mammo_roi/roi_centers_manifest.json`` and PNGs under
``data/mammo_roi/images/`` (basenames in ``src_imname``).

Keypoints match SPair JSON: list of ``[x, y]`` in PIL pixel coordinates on that
image file; the parent dataset rescales to 512×512 on load.
"""
from __future__ import annotations

import json
import os

import torch
from PIL import Image

from .dataset import CorrespondenceDataset


class MammoROIDataset(CorrespondenceDataset):
    def __init__(
        self,
        benchmark,
        datapath,
        thres,
        device,
        split,
        augmentation,
        feature_size,
        sub_class="all",
        item_index=-1,
    ):
        super(MammoROIDataset, self).__init__(
            benchmark, datapath, thres, device, split, augmentation, feature_size
        )
        manifest_path = os.path.join(datapath, "roi_centers_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run export_roi_centers_semantic_correspondence.py "
                "from StableKeypointsPlus with --semantic-dir pointing here."
            )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        anns = manifest.get("annotations") or []
        if item_index >= 0:
            anns = [anns[item_index]]
        self.train_data = [a.get("id", str(i)) for i, a in enumerate(anns)]
        self.src_imnames = [a["src_imname"] for a in anns]
        self.trg_imnames = [a.get("trg_imname", a["src_imname"]) for a in anns]
        self.cls = ["mammo_roi"]
        self.cls_ids = [0] * len(anns)
        self.src_kps = [torch.tensor(a["src_kps"]).t().float() for a in anns]
        self.trg_kps = [torch.tensor(a["trg_kps"]).t().float() for a in anns]

    def get_image(self, imnames, idx):
        name = imnames[idx]
        path = os.path.join(self.img_path, name)
        if not os.path.isfile(path):
            path = name
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        batch = super(MammoROIDataset, self).__getitem__(idx)
        batch["pckthres"] = torch.tensor([512.0])
        batch["idx"] = idx
        return batch
