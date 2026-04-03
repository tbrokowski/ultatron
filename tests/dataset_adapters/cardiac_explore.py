"""
cardiac_explore.py  ·  Real-data exploration for new cardiac datasets
======================================================================

Covers:  EchoNet-Pediatric, TED, Unity-Echo, MIMIC-IV-Echo-LVVol-A4C

Usage (from project root with .venv active):

    python -m tests.dataset_adapters.cardiac_explore

    # Run only selected adapters:
    US_SKIP_ECHONET_PED=1   python -m tests.dataset_adapters.cardiac_explore
    US_SKIP_TED=1           python -m tests.dataset_adapters.cardiac_explore
    US_SKIP_UNITY=1         python -m tests.dataset_adapters.cardiac_explore
    US_SKIP_MIMIC_LVVOL=1   python -m tests.dataset_adapters.cardiac_explore

Environment overrides (paths):
    US_ECHONET_PED_ROOT    Override EchoNet-Pediatric root
    US_TED_ROOT            Override TED root
    US_UNITY_ROOT          Override Unity root
    US_MIMIC_LVVOL_ROOT    Override MIMIC-IV-Echo-LVVol-A4C root

This script:
  1. Runs each adapter on the real dataset on Capstor.
  2. Builds a small manifest and prints statistics (entries, splits, labels).
  3. Loads raw frames / images and saves PNG snapshots.
  4. Applies VideoSSL / ImageSSL transforms and saves augmentation panels.
  5. Runs a DownstreamDataset smoke to confirm label delivery.

It is an exploratory tool and is NOT part of the automated pytest suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest, manifest_stats
from data.pipeline.transforms import (
    ImageSSLTransformConfig,
    VideoSSLTransformConfig,
    MASK_STRATEGY_FREQ,
    MASK_STRATEGY_SPATIAL,
    MASK_STRATEGY_BOTH,
)

# ── Capstor paths ─────────────────────────────────────────────────────────────
_STORE = Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac")

_DEFAULT_ROOTS = {
    "EchoNet-Pediatric":       _STORE / "EchoNet-Pediatric",
    "TED":                     _STORE / "TED",
    "Unity-Echo":              _STORE / "Unity",
    "MIMIC-IV-Echo-LVVol-A4C": _STORE / "MIMIC-IV-Echo-LVVol-A4C",
}

_ENV_VARS = {
    "EchoNet-Pediatric":       "US_ECHONET_PED_ROOT",
    "TED":                     "US_TED_ROOT",
    "Unity-Echo":              "US_UNITY_ROOT",
    "MIMIC-IV-Echo-LVVol-A4C": "US_MIMIC_LVVOL_ROOT",
}

_SKIP_ENV = {
    "EchoNet-Pediatric":       "US_SKIP_ECHONET_PED",
    "TED":                     "US_SKIP_TED",
    "Unity-Echo":              "US_SKIP_UNITY",
    "MIMIC-IV-Echo-LVVol-A4C": "US_SKIP_MIMIC_LVVOL",
}

_OUT_ROOT = Path("dataset_exploration_outputs")
N_VIS      = 6     # entries to visualise per dataset
N_PANELS   = 4     # entries for full augmentation panel
THUMB_SIZE = 112   # thumbnail size in panels
PAD        = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _root(ds_id: str) -> Optional[Path]:
    env_var = _ENV_VARS[ds_id]
    env     = os.environ.get(env_var)
    p       = Path(env) if env else _DEFAULT_ROOTS[ds_id]
    return p if p.exists() else None


def _frame_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim == 3:
        c = arr.shape[0] if arr.shape[0] in (1, 3) else None
        if c:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0].astype(np.uint8)).convert("RGB")
        return Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported shape: {arr.shape}")


def _tensor_frames(t) -> List[Image.Image]:
    import torch
    if isinstance(t, torch.Tensor):
        arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    else:
        arr = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    # (T, C, H, W) or (C, H, W)
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    return [_frame_to_pil(f) for f in arr]


def _thumb(img: Image.Image, label: str = "") -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (0, 0, 0))
    canvas.paste(img, ((THUMB_SIZE - img.width) // 2, (THUMB_SIZE - img.height) // 2))
    if label:
        bar  = Image.new("RGB", (THUMB_SIZE, 18), (30, 30, 30))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
        draw.text((3, 2), label[:22], fill=(220, 220, 220), font=font)
        full = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE + 18))
        full.paste(bar, (0, 0))
        full.paste(canvas, (0, 18))
        return full
    return canvas


def _build_panel(rows: List[List[Image.Image]]) -> Image.Image:
    cell_h = THUMB_SIZE + 18
    n_cols = max(len(r) for r in rows)
    canvas = Image.new(
        "RGB",
        (n_cols * (THUMB_SIZE + PAD) + PAD, len(rows) * (cell_h + PAD) + PAD),
        (20, 20, 20),
    )
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            canvas.paste(img, (PAD + ci * (THUMB_SIZE + PAD), PAD + ri * (cell_h + PAD)))
    return canvas


def _ensure(out_dir: Path, *subs: str) -> Path:
    for s in subs:
        (out_dir / s).mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_manifest(adapter, n_max: int, out_dir: Path, name: str) -> tuple[Path, List[USManifestEntry]]:
    path    = out_dir / f"{name}_manifest.jsonl"
    entries: List[USManifestEntry] = []
    for e in adapter.iter_entries():
        entries.append(e)
        if len(entries) >= n_max:
            break
    if not entries:
        raise RuntimeError(f"[{name.upper()}] No entries found.")
    with ManifestWriter(path) as w:
        for e in entries:
            w.write(e)
    print(f"[{name.upper()}] Wrote {len(entries)} entries → {path}")
    for k, v in manifest_stats(entries).items():
        print(f"  {k}: {v}")
    return path, entries


# ── Per-dataset loading helpers ───────────────────────────────────────────────

def _load_video(path: str) -> List[np.ndarray]:
    """Load video frames from .avi or .mhd cine sequence."""
    from data.pipeline.dataset import load_video_frames
    frames = load_video_frames(path)
    return frames or []


def _load_mhd_volume(path: str) -> List[np.ndarray]:
    """Load all frames from a 3-D .mhd volume (TED sequences)."""
    try:
        import SimpleITK as sitk
        img  = sitk.ReadImage(path)
        arr  = sitk.GetArrayFromImage(img)   # shape: (T, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        # Normalise to uint8
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]
    except ImportError:
        print("  [WARN] SimpleITK not installed — skipping .mhd visualisation")
        return []


def _load_image_pil(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EchoNet-Pediatric
# ══════════════════════════════════════════════════════════════════════════════

def explore_echonet_pediatric(out_root: Path) -> None:
    ds_id = "EchoNet-Pediatric"
    root  = _root(ds_id)
    if root is None:
        print(f"[ECHONET-PED] Root not found at {_DEFAULT_ROOTS[ds_id]} — skipping")
        return

    from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter
    from data.pipeline.transforms import VideoSSLTransform

    out = _ensure(out_root / "echonet_pediatric",
                  "raw_frames", "ssl_frames/full", "ssl_frames/visible",
                  "mask_debug", "summary_panels")

    _, entries = _build_manifest(EchoNetPediatricAdapter(root), 200, out, "echonet_ped")

    # ── Stats ──
    views: dict = {}
    efs: List[float] = []
    splits: dict = {}
    for e in entries:
        v = e.source_meta.get("view", "?")
        views[v]  = views.get(v, 0) + 1
        splits[e.split] = splits.get(e.split, 0) + 1
        ef = e.source_meta.get("ef", 0.0)
        if ef:
            efs.append(float(ef))
    print(f"[ECHONET-PED] Views: {views}")
    print(f"[ECHONET-PED] Splits: {splits}")
    if efs:
        print(f"[ECHONET-PED] EF — min={min(efs):.1f}  mean={mean(efs):.1f}  max={max(efs):.1f}")
    print(f"[ECHONET-PED] Entries with LV tracings: {sum(1 for e in entries if e.instances)}")

    # ── Raw frames ──
    for idx, e in enumerate(entries[:N_VIS]):
        frames = _load_video(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        for j, fi in enumerate([0, T // 2, max(T - 1, 0)]):
            _frame_to_pil(frames[fi]).save(
                out / "raw_frames" / f"ped_{idx:03d}_{e.source_meta.get('view','?')}_{j}.png"
            )
        print(f"[ECHONET-PED] Raw frames saved — idx={idx} view={e.source_meta.get('view')} split={e.split} ef={e.source_meta.get('ef',0.0):.1f}")

    # ── Augmentation panels ──
    vid_cfg = VideoSSLTransformConfig(
        n_frames=16, max_crop_px=224, min_crop_px=64, mask_strategy=MASK_STRATEGY_FREQ,
    )
    for idx, e in enumerate(entries[:N_PANELS]):
        frames = _load_video(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        raw_row = [
            _thumb(_frame_to_pil(frames[0]),        "raw first"),
            _thumb(_frame_to_pil(frames[T // 2]),   "raw mid"),
            _thumb(_frame_to_pil(frames[max(T-1,0)]), "raw last"),
        ]
        strategy_rows = []
        for tag, strat in (("freq", MASK_STRATEGY_FREQ), ("spatial", MASK_STRATEGY_SPATIAL), ("both", MASK_STRATEGY_BOTH)):
            cfg2   = VideoSSLTransformConfig(n_frames=16, max_crop_px=224, min_crop_px=64, mask_strategy=strat)
            views  = VideoSSLTransform(cfg2)(frames)
            full_f = _tensor_frames(views["full"])
            vis_f  = _tensor_frames(views["visible"])
            if full_f and vis_f:
                strategy_rows.append([_thumb(full_f[0], f"{tag} full"), _thumb(vis_f[0], f"{tag} vis")])
                full_f[0].save(out / "mask_debug" / f"ped_{idx:03d}_{tag}_full.png")
                vis_f[0].save(out / "mask_debug"  / f"ped_{idx:03d}_{tag}_vis.png")
        panel = _build_panel([raw_row] + strategy_rows)
        panel.save(out / "summary_panels" / f"ped_{idx:03d}_panel.png")
        print(f"[ECHONET-PED] Panel saved — idx={idx}")

    # ── Downstream label smoke ──
    from data.pipeline.downstream_dataset import DownstreamDataset
    cfg_ds = VideoSSLTransformConfig(n_frames=16, max_crop_px=224, min_crop_px=64)
    cfg_ds.global_crop_size = cfg_ds.max_crop_px  # type: ignore
    ds = DownstreamDataset(entries[:8], cfg=cfg_ds, training_mode="supervised")
    print(f"[ECHONET-PED] DownstreamDataset size: {len(ds)}")
    for i in range(min(3, len(ds))):
        item = ds[i]
        clip = item["image"]
        print(f"  sample {i}: clip.shape={tuple(clip.shape)} label_targets={item.get('label_targets')}")

    print("[ECHONET-PED] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# TED
# ══════════════════════════════════════════════════════════════════════════════

def explore_ted(out_root: Path) -> None:
    ds_id = "TED"
    root  = _root(ds_id)
    if root is None:
        print(f"[TED] Root not found at {_DEFAULT_ROOTS[ds_id]} — skipping")
        return

    from data.adapters.cardiac.ted import TEDAdapter
    from data.pipeline.transforms import VideoSSLTransform

    out = _ensure(out_root / "ted", "raw_frames", "mask_debug", "summary_panels")

    _, entries = _build_manifest(TEDAdapter(root), 200, out, "ted")

    video_entries = [e for e in entries if e.modality_type == "video"]
    image_entries = [e for e in entries if e.modality_type == "image"]

    efs   = [float(e.source_meta.get("ef", 0)) for e in video_entries if e.source_meta.get("ef")]
    quals = {}
    for e in video_entries:
        q = e.source_meta.get("image_quality", "?")
        quals[q] = quals.get(q, 0) + 1

    print(f"[TED] Video entries: {len(video_entries)}  Image (ED/ES) entries: {len(image_entries)}")
    if efs:
        print(f"[TED] EF — min={min(efs):.1f}  mean={mean(efs):.1f}  max={max(efs):.1f}  std={stdev(efs):.1f}")
    print(f"[TED] Image quality distribution: {quals}")
    print(f"[TED] Masked entries: {sum(1 for e in entries if e.has_mask)}")

    # ── Raw frames from MHD sequences ──
    for idx, e in enumerate(video_entries[:N_VIS]):
        frames = _load_mhd_volume(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        for j, fi in enumerate([0, T // 2, max(T - 1, 0)]):
            _frame_to_pil(frames[fi]).save(
                out / "raw_frames" / f"ted_{idx:03d}_seq_{j}.png"
            )
        # Also save ED and ES thumbnails from source_meta
        ed_fi = e.source_meta.get("ed_frame", 0)
        es_fi = e.source_meta.get("es_frame", 0)
        if ed_fi and ed_fi < len(frames):
            _frame_to_pil(frames[ed_fi]).save(out / "raw_frames" / f"ted_{idx:03d}_ED.png")
        if es_fi and es_fi < len(frames):
            _frame_to_pil(frames[es_fi]).save(out / "raw_frames" / f"ted_{idx:03d}_ES.png")
        print(f"[TED] Raw frames — idx={idx} patient={e.study_id} nframes={T} ef={e.source_meta.get('ef',0):.1f}")

    # ── Augmentation panels (MHD → VideoSSL) ──
    for idx, e in enumerate(video_entries[:N_PANELS]):
        frames = _load_mhd_volume(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        raw_row = [
            _thumb(_frame_to_pil(frames[0]),          "raw first"),
            _thumb(_frame_to_pil(frames[T // 2]),     "raw mid"),
            _thumb(_frame_to_pil(frames[max(T-1,0)]), "raw last"),
        ]
        strategy_rows = []
        for tag, strat in (("freq", MASK_STRATEGY_FREQ), ("spatial", MASK_STRATEGY_SPATIAL)):
            cfg2  = VideoSSLTransformConfig(n_frames=min(16, T), max_crop_px=224, min_crop_px=64, mask_strategy=strat)
            views = VideoSSLTransform(cfg2)(frames)
            full_f = _tensor_frames(views["full"])
            vis_f  = _tensor_frames(views["visible"])
            if full_f and vis_f:
                strategy_rows.append([_thumb(full_f[0], f"{tag} full"), _thumb(vis_f[0], f"{tag} vis")])
                full_f[0].save(out / "mask_debug" / f"ted_{idx:03d}_{tag}_full.png")
        _build_panel([raw_row] + strategy_rows).save(out / "summary_panels" / f"ted_{idx:03d}_panel.png")
        print(f"[TED] Panel saved — idx={idx}")

    print("[TED] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Unity
# ══════════════════════════════════════════════════════════════════════════════

def explore_unity(out_root: Path) -> None:
    ds_id = "Unity-Echo"
    root  = _root(ds_id)
    if root is None:
        print(f"[UNITY] Root not found at {_DEFAULT_ROOTS[ds_id]} — skipping")
        return

    from data.adapters.cardiac.unity import UnityAdapter
    from data.pipeline.transforms import ImageSSLTransform

    out = _ensure(out_root / "unity", "raw_images", "ssl_crops", "summary_panels")

    _, entries = _build_manifest(UnityAdapter(root), 300, out, "unity")

    train_entries = [e for e in entries if e.split == "train"]
    tune_entries  = [e for e in entries if e.split == "val"]
    active_kp_counts = [len(e.source_meta.get("keypoints", {})) for e in entries]
    print(f"[UNITY] Train: {len(train_entries)}  Tune/Val: {len(tune_entries)}")
    print(f"[UNITY] Active keypoints per frame — min={min(active_kp_counts)}  mean={mean(active_kp_counts):.1f}  max={max(active_kp_counts)}")

    # ── Raw images + keypoint overlay ──
    for idx, e in enumerate(entries[:N_VIS]):
        img = _load_image_pil(e.image_paths[0])
        if img is None:
            continue
        img_kp = img.copy()
        draw   = ImageDraw.Draw(img_kp)
        for kp_name, kp in e.source_meta.get("keypoints", {}).items():
            kp_type = kp.get("type", "")
            if kp_type == "point" and kp.get("x") and kp.get("y"):
                try:
                    x, y = float(kp["x"]), float(kp["y"])
                    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(255, 80, 80))
                except ValueError:
                    pass
            elif kp_type == "curve" and kp.get("x") and kp.get("y"):
                try:
                    xs = [float(v) for v in kp["x"].split()]
                    ys = [float(v) for v in kp["y"].split()]
                    pts = list(zip(xs, ys))
                    if len(pts) >= 2:
                        draw.line(pts, fill=(80, 200, 80), width=2)
                except (ValueError, Exception):
                    pass
        img.save(out / "raw_images" / f"unity_{idx:03d}_raw.png")
        img_kp.save(out / "raw_images" / f"unity_{idx:03d}_kp.png")
        print(f"[UNITY] Image saved — idx={idx} active_kp={len(e.source_meta.get('keypoints',{}))}")

    # ── ImageSSL augmentation panels ──
    cfg = ImageSSLTransformConfig(
        n_global_crops=2, n_local_crops=4,
        max_global_crop_px=224, min_crop_px=32, mask_strategy=MASK_STRATEGY_FREQ,
    )
    for idx, e in enumerate(entries[:N_PANELS]):
        img = _load_image_pil(e.image_paths[0])
        if img is None:
            continue
        views    = ImageSSLTransform(cfg)(img)
        g_crops  = views.get("global_crops", [])
        l_crops  = views.get("local_crops",  [])
        row0 = [_thumb(img, "original")]
        row1 = [_thumb(t, f"global_{i}") for i, t in enumerate(_tensor_frames(g_crops[0]) if g_crops else [])]
        row2 = [_thumb(t, f"local_{i}")  for i, t in enumerate(_tensor_frames(l_crops[0])  if l_crops  else [][:4])]
        _build_panel([row0, row1, row2]).save(out / "summary_panels" / f"unity_{idx:03d}_panel.png")
        # Save individual SSL crops for inspection
        for i, gc in enumerate(g_crops or []):
            gf = _tensor_frames(gc)
            if gf:
                gf[0].save(out / "ssl_crops" / f"unity_{idx:03d}_global_{i}.png")
        print(f"[UNITY] Panel saved — idx={idx}")

    # ── Downstream smoke: confirm keypoints reach sample dict ──
    from data.pipeline.downstream_dataset import DownstreamDataset
    cfg_ds = ImageSSLTransformConfig(n_global_crops=2, n_local_crops=0, max_global_crop_px=224, min_crop_px=64)
    cfg_ds.global_crop_size = cfg_ds.max_global_crop_px  # type: ignore
    ds = DownstreamDataset(entries[:8], cfg=cfg_ds, training_mode="supervised")
    print(f"[UNITY] DownstreamDataset size: {len(ds)}")
    for i in range(min(3, len(ds))):
        item = ds[i]
        kp   = item.get("source_meta", {}).get("keypoints", {})
        img  = item.get("image")
        print(f"  sample {i}: img.shape={tuple(img.shape) if hasattr(img,'shape') else '?'} kp_count={len(kp)}")

    print("[UNITY] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# MIMIC-IV-Echo-LVVol-A4C
# ══════════════════════════════════════════════════════════════════════════════

def explore_mimic_lvvol(out_root: Path) -> None:
    ds_id = "MIMIC-IV-Echo-LVVol-A4C"
    root  = _root(ds_id)
    if root is None:
        print(f"[MIMIC-LVVOL] Root not found at {_DEFAULT_ROOTS[ds_id]} — skipping")
        return

    from data.adapters.cardiac.mimic_lvvol_a4c import MIMICLVVolA4CAdapter
    from data.pipeline.transforms import VideoSSLTransform

    out = _ensure(out_root / "mimic_lvvol", "raw_frames", "mask_debug", "summary_panels")

    _, entries = _build_manifest(MIMICLVVolA4CAdapter(root), 100, out, "mimic_lvvol")

    efs_a4c = [float(e.source_meta.get("lvef_a4c", float("nan"))) for e in entries]
    efs_a4c = [v for v in efs_a4c if not __import__("math").isnan(v)]
    manuf   = {}
    for e in entries:
        m = e.source_meta.get("manufacturer", "?")
        manuf[m] = manuf.get(m, 0) + 1
    print(f"[MIMIC-LVVOL] Entries: {len(entries)}")
    if efs_a4c:
        print(f"[MIMIC-LVVOL] LVEF_A4C — min={min(efs_a4c):.1f}  mean={mean(efs_a4c):.1f}  max={max(efs_a4c):.1f}")
    print(f"[MIMIC-LVVOL] Manufacturers: {manuf}")

    # ── Load DICOM frames and save raw PNGs ──
    try:
        import pydicom
        import cv2
        _dcm_ok = True
    except ImportError:
        _dcm_ok = False
        print("[MIMIC-LVVOL] pydicom or cv2 not installed — skipping frame visualisation")

    def _load_dicom_frames(dcm_path: str) -> List[np.ndarray]:
        if not _dcm_ok:
            return []
        try:
            ds_dcm = pydicom.dcmread(dcm_path)
            arr    = ds_dcm.pixel_array   # (T, H, W, C) or (T, H, W) or (H, W)
            if arr.ndim == 2:
                return [arr]
            if arr.ndim == 3:
                if arr.shape[-1] in (3, 4):   # single frame with channels
                    return [arr[:, :, :3]]
                return [arr[i] for i in range(arr.shape[0])]
            if arr.ndim == 4:                  # (T, H, W, C)
                return [arr[i, :, :, :3] for i in range(arr.shape[0])]
            return []
        except Exception as ex:
            print(f"  [WARN] Failed to load {dcm_path}: {ex}")
            return []

    for idx, e in enumerate(entries[:N_VIS]):
        frames = _load_dicom_frames(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        for j, fi in enumerate([0, T // 2, max(T - 1, 0)]):
            f = frames[fi]
            pil = _frame_to_pil(f)
            pil.save(out / "raw_frames" / f"lvvol_{idx:03d}_{j}.png")
        print(f"[MIMIC-LVVOL] Frames saved — idx={idx} study_id={e.study_id}"
              f" lvef={e.source_meta.get('lvef_a4c',float('nan')):.1f}")

    # ── VideoSSL augmentation panels from DICOM frames ──
    for idx, e in enumerate(entries[:N_PANELS]):
        frames = _load_dicom_frames(e.image_paths[0])
        if not frames:
            continue
        T = len(frames)
        raw_row = [
            _thumb(_frame_to_pil(frames[0]),          "raw first"),
            _thumb(_frame_to_pil(frames[T // 2]),     "raw mid"),
            _thumb(_frame_to_pil(frames[max(T-1,0)]), "raw last"),
        ]
        strategy_rows = []
        for tag, strat in (("freq", MASK_STRATEGY_FREQ), ("spatial", MASK_STRATEGY_SPATIAL)):
            cfg2  = VideoSSLTransformConfig(n_frames=min(16, T), max_crop_px=224, min_crop_px=64, mask_strategy=strat)
            views = VideoSSLTransform(cfg2)(frames)
            full_f = _tensor_frames(views["full"])
            vis_f  = _tensor_frames(views["visible"])
            if full_f and vis_f:
                strategy_rows.append([_thumb(full_f[0], f"{tag} full"), _thumb(vis_f[0], f"{tag} vis")])
                full_f[0].save(out / "mask_debug" / f"lvvol_{idx:03d}_{tag}_full.png")
        if strategy_rows:
            _build_panel([raw_row] + strategy_rows).save(
                out / "summary_panels" / f"lvvol_{idx:03d}_panel.png"
            )
            print(f"[MIMIC-LVVOL] Panel saved — idx={idx}")

    print("[MIMIC-LVVOL] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    out_root = Path(os.environ.get("US_CARDIAC_EXPLORE_OUT", str(_OUT_ROOT)))
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Cardiac Dataset Exploration")
    print(f" Output root: {out_root.resolve()}")
    print("=" * 70)

    if not os.environ.get("US_SKIP_ECHONET_PED"):
        explore_echonet_pediatric(out_root)
    if not os.environ.get("US_SKIP_TED"):
        explore_ted(out_root)
    if not os.environ.get("US_SKIP_UNITY"):
        explore_unity(out_root)
    if not os.environ.get("US_SKIP_MIMIC_LVVOL"):
        explore_mimic_lvvol(out_root)

    print("\n" + "=" * 70)
    print(" Cardiac exploration complete.")
    print(f" All outputs under: {out_root.resolve()}/")
    print("   echonet_pediatric/    raw_frames/ mask_debug/ summary_panels/")
    print("   ted/                  raw_frames/ mask_debug/ summary_panels/")
    print("   unity/                raw_images/ ssl_crops/  summary_panels/")
    print("   mimic_lvvol/          raw_frames/ mask_debug/ summary_panels/")
    print("=" * 70)


if __name__ == "__main__":
    main()
