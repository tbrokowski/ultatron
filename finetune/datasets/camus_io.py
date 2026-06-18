"""
finetune/datasets/camus_io.py  ·  CAMUS split + mask protocol helpers
====================================================================

References
----------
Leclerc et al., "Deep Learning for Segmentation using an Open Large-Scale
Dataset in 2D Echocardiography", IEEE TMI 2019.
https://doi.org/10.1109/TMI.2019.2900516

Official evaluation protocol (TMI 2019)
---------------------------------------
* **10-fold cross-validation**: 500 patients in 10 balanced folds of 50.
  For each fold: 8 folds (400 patients) train, 1 fold (50) validation,
  1 fold (50) test.
* **Structures** (reported separately): LVEndo (endocardium, label 1) and
  LVEpi (epicardium = cavity + myocardium, labels 1+2). LA is label 3.
* **Frames**: 2CH and 4CH apical views at ED and ES only.
* **Metrics**: Dice, mean absolute distance (dm), Hausdorff (dH) per structure.
* **Quality filter** (Table III): many papers evaluate on Good + Medium quality
  only (~406/450 training patients per fold).

Fixed split used here (single-fold instance of TMI CV)
------------------------------------------------------
When folds are assigned by consecutive patient-ID blocks of 50:

  train : patient0001 – patient0400  (folds 1–8)
  val   : patient0401 – patient0450  (fold 9)
  test  : patient0451 – patient0500  (fold 10)

All frames from one patient (2CH/4CH × ED/ES) share the same split.
"""
from __future__ import annotations

import configparser
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

log = logging.getLogger(__name__)

# Inclusive patient ID ranges — equivalent to TMI folds 1–8 / 9 / 10 by ID block.
OFFICIAL_CAMUS_SPLIT = {
    "train": (1, 400),
    "val": (401, 450),
    "test": (451, 500),
}

FOLD_SIZE = 50
N_FOLDS = 10

LvTarget = Literal["lv_cavity", "lv_myocardium", "lv_structures", "lv_la", "any_foreground"]
CamusTrainingMode = Literal["binary", "multiclass"]
CamusVariant = Literal["lv_endo", "lv_epi", "la", "multiclass"]

# TMI 2019 paper names ↔ our lv_target keys
LV_TARGET_ALIASES = {
    "lv_cavity": "LVEndo",
    "lv_structures": "LVEpi",
    "lv_myocardium": "LV_myocardium",
    "lv_la": "LA",
}

_LV_TARGET_DOC = {
    "lv_cavity": "label==1 (LVEndo / LV endocardium cavity)",
    "lv_myocardium": "label==2 (LV myocardium between endo and epi)",
    "lv_structures": "labels 1+2 (LVEpi / cavity + myocardium; excludes LA)",
    "lv_la": "label==3 (left atrium)",
    "any_foreground": "any non-zero label (legacy; not TMI protocol)",
}

CAMUS_VARIANTS: dict[str, dict] = {
    "lv_endo": {
        "mode": "binary",
        "lv_target": "lv_cavity",
        "dice_key": "dice_lv_endo",
        "metric_prefix": "dice_lv_endo",
        "paper_name": "LVEndo",
    },
    "lv_epi": {
        "mode": "binary",
        "lv_target": "lv_structures",
        "dice_key": "dice_lv_epi",
        "metric_prefix": "dice_lv_epi",
        "paper_name": "LVEpi",
    },
    "la": {
        "mode": "binary",
        "lv_target": "lv_la",
        "dice_key": "dice_la",
        "metric_prefix": "dice_la",
        "paper_name": "LA",
    },
    "multiclass": {
        "mode": "multiclass",
        "lv_target": "lv_structures",
        "dice_keys": ("dice_class_1", "dice_class_2", "dice_class_3"),
        "metric_prefix": "dice_class",
        "paper_name": "multiclass",
    },
}

DEFAULT_CAMUS_VARIANT_LIST = [
    {"variant": "lv_endo", "camus_training_mode": "binary", "lv_target": "lv_cavity"},
    {"variant": "lv_epi", "camus_training_mode": "binary", "lv_target": "lv_structures"},
    {"variant": "la", "camus_training_mode": "binary", "lv_target": "lv_la"},
    {"variant": "multiclass", "camus_training_mode": "multiclass"},
]


def get_camus_variant_spec(variant: str) -> dict:
    if variant not in CAMUS_VARIANTS:
        raise ValueError(f"Unknown camus_variant={variant!r}")
    return CAMUS_VARIANTS[variant]

ACCEPTABLE_IMAGE_QUALITIES = frozenset({"good", "medium"})


def parse_patient_id(patient_dir_name: str) -> int | None:
    """Extract numeric ID from ``patient0184`` → 184."""
    m = re.match(r"patient0*(\d+)$", patient_dir_name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def fold_for_patient(patient_id: int) -> int | None:
    """Return TMI fold index 1–10 for a patient ID (50 patients per fold)."""
    if patient_id < 1 or patient_id > 500:
        return None
    return (patient_id - 1) // FOLD_SIZE + 1


def official_split_for_patient(patient_id: int) -> str | None:
    """Return train/val/test for fixed fold-10-test protocol, or None if out of range."""
    fold = fold_for_patient(patient_id)
    if fold is None:
        return None
    if fold <= 8:
        return "train"
    if fold == 9:
        return "val"
    return "test"


def read_image_quality(patient_dir: Path, view: str) -> str | None:
    """Parse ``ImageQuality`` from ``Info_{view}.cfg`` (Good / Medium / Poor)."""
    cfg_path = patient_dir / f"Info_{view}.cfg"
    if not cfg_path.is_file():
        return None
    try:
        text = "[info]\n" + cfg_path.read_text()
        parser = configparser.ConfigParser()
        parser.read_string(text)
        raw = parser["info"].get("imagequality", "").strip()
        return raw or None
    except (configparser.Error, OSError):
        return None


def quality_is_acceptable(quality: str | None) -> bool:
    """TMI Table III uses Good + Medium quality patients only."""
    if quality is None:
        return True
    return quality.strip().lower() in ACCEPTABLE_IMAGE_QUALITIES


def load_camus_slice(path: str | Path) -> np.ndarray:
    """Load a CAMUS .mhd / .nii.gz volume and return a 2-D float32 slice."""
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if arr.ndim == 3:
        # ED/ES exports are single-slice volumes (Z=1); take index 0.
        arr = arr[0]
    return arr


def encode_lv_mask(
    label_map: np.ndarray,
    target: LvTarget = "lv_structures",
) -> np.ndarray:
    """
    Convert CAMUS integer label map to binary float mask.

    CAMUS GT labels (database_nifti):
      0 = background
      1 = LV cavity (LVEndo)
      2 = LV myocardium
      3 = left atrium (when present)
    """
    arr = label_map.astype(np.int64)
    if target == "lv_cavity":
        return (arr == 1).astype(np.float32)
    if target == "lv_myocardium":
        return (arr == 2).astype(np.float32)
    if target == "lv_structures":
        return ((arr == 1) | (arr == 2)).astype(np.float32)
    if target == "lv_la":
        return (arr == 3).astype(np.float32)
    if target == "any_foreground":
        return (arr > 0).astype(np.float32)
    raise ValueError(f"Unknown lv_target={target!r}")


@dataclass(frozen=True)
class CamusFrameSample:
    img: str
    msk: str
    sample_id: str
    patient: str
    patient_id: int
    view: str
    phase: str
    image_quality: str | None


def iter_camus_frames(
    root: Path,
    split: str,
    *,
    lv_target: LvTarget = "lv_structures",
    quality_filter: bool = False,
) -> Iterator[CamusFrameSample]:
    """
    Yield ED/ES frame paths for one split using the official patient protocol.
    """
    nifti_dir = root / "database_nifti"
    pdir_root = nifti_dir if nifti_dir.exists() else root

    ext = ".nii.gz" if any(pdir_root.rglob("*.nii.gz")) else ".mhd"
    gt_sfx = f"_gt{ext}"

    for pdir in sorted(pdir_root.glob("patient*/")):
        pid_num = parse_patient_id(pdir.name)
        if pid_num is None:
            continue
        if official_split_for_patient(pid_num) != split:
            continue

        pid = pdir.name
        for view in ("2CH", "4CH"):
            quality = read_image_quality(pdir, view)
            if quality_filter and not quality_is_acceptable(quality):
                continue
            for phase in ("ED", "ES"):
                img = pdir / f"{pid}_{view}_{phase}{ext}"
                msk = pdir / f"{pid}_{view}_{phase}{gt_sfx}"
                if img.is_file() and msk.is_file():
                    yield CamusFrameSample(
                        img=str(img),
                        msk=str(msk),
                        sample_id=f"{pid}_{view}_{phase}",
                        patient=pid,
                        patient_id=pid_num,
                        view=view,
                        phase=phase,
                        image_quality=quality,
                    )


def collect_camus_frames(
    root: str | Path,
    split: str,
    *,
    lv_target: LvTarget = "lv_structures",
    quality_filter: bool = False,
) -> list[CamusFrameSample]:
    samples = list(
        iter_camus_frames(
            Path(root), split, lv_target=lv_target, quality_filter=quality_filter,
        )
    )
    log.info(
        "CAMUS %s: %d frames (lv_target=%s, quality_filter=%s, official patient split)",
        split,
        len(samples),
        lv_target,
        quality_filter,
    )
    return samples


def verify_camus_protocol(
    patients: list[str],
    *,
    lv_target: LvTarget = "lv_structures",
    quality_filter: bool = False,
) -> dict:
    """
    Log split integrity checks before training/evaluation.

    Verifies patient-level official split, fold assignment, no duplicate
    patients across splits, and documents the binary target definition.
    """
    by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    by_fold: dict[int, set[int]] = {i: set() for i in range(1, N_FOLDS + 1)}
    unknown: list[str] = []

    for pname in patients:
        pid = parse_patient_id(pname)
        if pid is None:
            unknown.append(pname)
            continue
        split = official_split_for_patient(pid)
        fold = fold_for_patient(pid)
        if split is None or fold is None:
            unknown.append(pname)
            continue
        by_split[split].add(pid)
        by_fold[fold].add(pid)

    overlap = []
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            inter = by_split[a] & by_split[b]
            if inter:
                overlap.append((a, b, sorted(inter)[:5]))

    report = {
        "split_policy": "tmi2019_fixed_fold10_test",
        "split_equivalent_to": "folds 1-8 train, fold 9 val, fold 10 test (50 patients each)",
        "lv_target": lv_target,
        "lv_target_paper_name": LV_TARGET_ALIASES.get(lv_target, lv_target),
        "lv_target_definition": _LV_TARGET_DOC.get(lv_target, lv_target),
        "quality_filter": quality_filter,
        "quality_filter_definition": "Good + Medium only (TMI Table III)" if quality_filter else "all qualities",
        "n_patients": {k: len(v) for k, v in by_split.items()},
        "n_frames": {k: len(v) * 4 for k, v in by_split.items()},
        "n_folds_represented": sum(1 for f, pts in by_fold.items() if pts),
        "unknown_patients": unknown[:10],
        "patient_leakage": overlap,
    }

    log.info(
        "[CAMUS] Protocol: %s | target=%s (%s, paper: %s)",
        report["split_policy"],
        lv_target,
        report["lv_target_definition"],
        report["lv_target_paper_name"],
    )
    for split in ("train", "val", "test"):
        lo, hi = OFFICIAL_CAMUS_SPLIT[split]
        log.info(
            "  %s: %d patients (IDs %d–%d), ~%d frames",
            split,
            report["n_patients"][split],
            lo,
            hi,
            report["n_frames"][split],
        )
    if quality_filter:
        log.info("  quality: excluding Poor (TMI Table III good+medium only)")
    if unknown:
        log.warning("[CAMUS] %d patient dir(s) outside official ID range 1–500", len(unknown))
    if overlap:
        log.error("[CAMUS] Patient leakage across splits: %s", overlap)
    else:
        log.info("[CAMUS] No patient-level split leakage detected.")

    return report


def _aggregate_metric_block(
    per_sample: list[dict],
    dice_key: str,
    prefix: str,
) -> dict:
    def _mean(key: str, subset: list[dict] | None = None) -> float:
        rows = subset if subset is not None else per_sample
        vals = [s[key] for s in rows if key in s and not np.isnan(s[key])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    def _mean_subset(view: str, phase: str, key: str) -> float:
        sub = [s for s in per_sample if s.get("view") == view and s.get("phase") == phase]
        return _mean(key, sub)

    def _pool_phase(phase: str) -> float:
        sub = [s for s in per_sample if s.get("phase") == phase]
        return _mean(dice_key, sub)

    out = {
        f"{prefix}_mean": _mean(dice_key),
        "iou_mean": _mean("iou"),
        "hd95_mean": _mean("hd95"),
        f"{prefix}_2ch_ed": _mean_subset("2CH", "ED", dice_key),
        f"{prefix}_2ch_es": _mean_subset("2CH", "ES", dice_key),
        f"{prefix}_4ch_ed": _mean_subset("4CH", "ED", dice_key),
        f"{prefix}_4ch_es": _mean_subset("4CH", "ES", dice_key),
        f"{prefix}_ed": _pool_phase("ED"),
        f"{prefix}_es": _pool_phase("ES"),
    }
    out["dice_mean"] = out[f"{prefix}_mean"]
    out["dice_2ch_ed"] = out[f"{prefix}_2ch_ed"]
    out["dice_2ch_es"] = out[f"{prefix}_2ch_es"]
    out["dice_4ch_ed"] = out[f"{prefix}_4ch_ed"]
    out["dice_4ch_es"] = out[f"{prefix}_4ch_es"]
    return out


def aggregate_camus_metrics(per_sample: list[dict], variant: str) -> dict:
    """Aggregate paired metrics for a single CAMUS training variant."""
    spec = get_camus_variant_spec(variant)
    out: dict = {"camus_variant": variant, "camus_training_mode": spec["mode"]}

    if spec["mode"] == "binary":
        out.update(_aggregate_metric_block(per_sample, spec["dice_key"], spec["metric_prefix"]))
        out["lv_target"] = spec["lv_target"]
        return out

    for cls_idx, dice_key in enumerate(spec["dice_keys"], start=1):
        out.update(_aggregate_metric_block(per_sample, dice_key, f"dice_class_{cls_idx}"))

    fg_means = [out[f"dice_class_{i}_mean"] for i in (1, 2, 3)]
    valid = [v for v in fg_means if not np.isnan(v)]
    out["dice_macro_fg_mean"] = round(float(np.mean(valid)), 4) if valid else float("nan")
    out["dice_mean"] = out["dice_macro_fg_mean"]
    return out


def aggregate_structure_metrics(per_sample: list[dict], variant: str = "lv_epi") -> dict:
    return aggregate_camus_metrics(per_sample, variant)
