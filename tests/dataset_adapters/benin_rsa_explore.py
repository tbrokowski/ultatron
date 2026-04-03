"""
benin_rsa_explore.py  ·  Exploration for Benin-LUS and RSA-LUS adapters
=======================================================================

Usage (from project root):

    python -m tests.dataset_adapters.benin_rsa_explore

This script:
  1. Uses BeninLUSAdapter and RSALUSAdapter on the real CSCS data.
  2. Writes small manifests for each dataset.
  3. Prints basic stats and label distributions.
  4. Saves raw frames from a few clips.
  5. Runs a tiny DownstreamDataset to show label_targets, including:
       - lus_video_multilabel
       - lus_patient_tb / lus_patient_pneumonia / lus_patient_covid (via PatientLevelDataset)

It is intended for manual, interactive inspection only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from data.adapters.lung.benin_lus import BeninLUSAdapter
from data.adapters.lung.rsa_lus import RSALUSAdapter
from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest, manifest_stats
from data.pipeline.dataset import load_video_frames
from data.pipeline.downstream_dataset import DownstreamDataset, PatientLevelDataset
from data.pipeline.transforms import ImageSSLTransformConfig, VideoSSLTransformConfig


DEFAULT_BENIN_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Benin_Videos"
)
DEFAULT_RSA_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/RSA_Videos"
)
DEFAULT_OUT_ROOT = Path("dataset_exploration_outputs")

N_VIS_CLIPS = 6


def _frame_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0].astype(np.uint8)).convert("RGB")
        return Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unexpected frame shape: {arr.shape}")


def get_roots() -> tuple[Path, Path]:
    benin_env = os.environ.get("US_BENIN_ROOT")
    rsa_env = os.environ.get("US_RSA_ROOT")
    benin_root = Path(benin_env) if benin_env else DEFAULT_BENIN_ROOT
    rsa_root = Path(rsa_env) if rsa_env else DEFAULT_RSA_ROOT
    if not benin_root.exists():
        raise FileNotFoundError(f"Benin-LUS root not found at {benin_root}")
    if not rsa_root.exists():
        raise FileNotFoundError(f"RSA-LUS root not found at {rsa_root}")
    return benin_root, rsa_root


def ensure_out_dirs() -> tuple[Path, Path]:
    base = DEFAULT_OUT_ROOT
    benin_out = base / "benin"
    rsa_out = base / "rsa"
    for d in (benin_out, rsa_out):
        (d / "raw_frames").mkdir(parents=True, exist_ok=True)
    return benin_out, rsa_out


def build_manifest(adapter, out_dir: Path, name: str) -> Path:
    manifest_path = out_dir / f"{name}_manifest.jsonl"
    entries: List[USManifestEntry] = list(adapter.iter_entries())
    if not entries:
        raise RuntimeError(f"No entries produced by {adapter.__class__.__name__}")
    with ManifestWriter(manifest_path) as w:
        for e in entries:
            w.write(e)
    stats = manifest_stats(entries)
    print(f"[{name}] Wrote {len(entries)} entries to {manifest_path}")
    print(f"[{name}] Manifest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # Patient label distribution
    tb = pneu = cov = 0
    for e in entries:
        pl = e.source_meta.get("patient_labels") or {}
        tb += int(pl.get("tb", 0))
        pneu += int(pl.get("pneumonia", 0))
        cov += int(pl.get("covid", 0))
    print(f"[{name}] Patient labels: TB={tb}, Pneumonia={pneu}, Covid={cov}")
    return manifest_path


def save_raw_frames(entries: List[USManifestEntry], out_dir: Path, name: str) -> None:
    for idx, e in enumerate(entries[:N_VIS_CLIPS]):
        frames = load_video_frames(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        sel = [0, T // 2, max(T - 1, 0)]
        for j, fi in enumerate(sel):
            f = _frame_to_pil(frames[fi])
            f.save(out_dir / "raw_frames" / f"{name}_{idx:04d}_{j}.png")
        print(f"[{name}] Saved raw frames for idx={idx} id={e.sample_id}")


def downstream_smoke(manifest_path: Path, dataset_ids: List[str], name: str) -> None:
    print(f"[{name}] DownstreamDataset smoke:")
    entries = load_manifest(manifest_path, split="train")
    entries = [e for e in entries if e.dataset_id in dataset_ids]
    ds = DownstreamDataset(entries[:8], cfg=ImageSSLTransformConfig())
    for i in range(min(4, len(ds))):
        item = ds[i]
        print(f"  sample {i}: image.shape={tuple(item['image'].shape)} "
              f"n_targets={len(item['label_targets'])}")
        for t in item["label_targets"]:
            print(f"    head={t.head_id} type={t.head_type} "
                  f"shape={None if t.value is None else tuple(t.value.shape)}")

    print(f"[{name}] PatientLevelDataset smoke:")
    pds = PatientLevelDataset(
        entries[:32],
        cfg=ImageSSLTransformConfig(),
        video_cfg=VideoSSLTransformConfig(n_frames=4),
    )
    for i in range(min(4, len(pds))):
        item = pds[i]
        tmask = item.get("tube_mask")
        print(f"  study {i}: frames.shape={tuple(item['frames'].shape)} "
              f"tube_mask={None if tmask is None else tuple(tmask.shape)} "
              f"n_targets={len(item['label_targets'])}")
        for t in item["label_targets"]:
            print(f"    head={t.head_id} type={t.head_type} "
                  f"value={t.value.item() if t.value is not None else None}")


def main() -> None:
    benin_root, rsa_root = get_roots()
    benin_out, rsa_out = ensure_out_dirs()

    print(f"[BENIN] root: {benin_root}")
    print(f"[RSA]   root: {rsa_root}")
    print(f"[OUT]   base: {DEFAULT_OUT_ROOT.resolve()}")

    benin_manifest = build_manifest(BeninLUSAdapter(benin_root), benin_out, "benin_lus")
    rsa_manifest = build_manifest(RSALUSAdapter(rsa_root), rsa_out, "rsa_lus")

    benin_entries = load_manifest(benin_manifest, split="train")
    rsa_entries = load_manifest(rsa_manifest, split="train")

    save_raw_frames(benin_entries, benin_out, "benin")
    save_raw_frames(rsa_entries, rsa_out, "rsa")

    downstream_smoke(benin_manifest, ["Benin-LUS"], "Benin-LUS")
    downstream_smoke(rsa_manifest, ["RSA-LUS"], "RSA-LUS")

    print("[benin_rsa_explore] Complete.")


if __name__ == "__main__":
    main()

