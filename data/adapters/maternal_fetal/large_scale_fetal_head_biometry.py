"""
data/adapters/maternal_fetal/large_scale_fetal_head_biometry.py
================================================================

Large-Scale Fetal Head Biometry dataset (Alzubaidi et al., 2023).

The dataset combines FETAL-PLANES-DB fetal-brain planes with reannotated
HC18 fetal-head images.  We use the resized fixed-size RGB image folders as
the primary image source and ignore the "orginal-size" folders, which are
variable-size RGBA copies intended for workflows that need original geometry.

Annotation folders are already extracted in the dataset copy used for training,
so this adapter indexes extracted segmentation-mask PNGs directly and ignores
the original-size image folders.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


@dataclass(frozen=True)
class PlaneGroup:
    group_dir: str
    image_subdir: str
    pixel_csv: Optional[str]
    plane_label: str
    plane_index: int
    source_subset: str


@dataclass(frozen=True)
class StructureSpec:
    key: str
    label_raw: str
    label_ontology: str
    mask_channel: int
    geometry_type: str


PLANE_GROUPS: Sequence[PlaneGroup] = (
    PlaneGroup(
        group_dir="Trans-thalamic",
        image_subdir="Trans-thalamic",
        pixel_csv="Trans-Thalamic-Pixel-Size.csv",
        plane_label="Trans-thalamic",
        plane_index=0,
        source_subset="FETAL_PLANES_DB",
    ),
    PlaneGroup(
        group_dir="Trans-cerebellum",
        image_subdir="Trans-cerebellum",
        pixel_csv="Trans-cerebellum-Pixel-Size.csv",
        plane_label="Trans-cerebellum",
        plane_index=1,
        source_subset="FETAL_PLANES_DB",
    ),
    PlaneGroup(
        group_dir="Trans-ventricular",
        image_subdir="Trans-ventricular",
        pixel_csv="Trans-ventricular-Pixel-Size.csv",
        plane_label="Trans-ventricular",
        plane_index=2,
        source_subset="FETAL_PLANES_DB",
    ),
    PlaneGroup(
        group_dir="Diverse Fetal Head Images",
        image_subdir="Orginal_train_images_to_959_661",
        pixel_csv=None,
        plane_label="Diverse fetal head",
        plane_index=3,
        source_subset="HC18",
    ),
)

STRUCTURES: Sequence[StructureSpec] = (
    StructureSpec("brain", "Brain", "fetal_brain", 0, "ellipse"),
    StructureSpec("csp", "CSP", "cavum_septi_pellucidi", 1, "oriented_rectangle"),
    StructureSpec("lv", "LV", "lateral_ventricle", 2, "oriented_rectangle"),
)

_PATIENT_RE = re.compile(r"Patient(\d+)", re.IGNORECASE)
_NUMERIC_PREFIX_RE = re.compile(r"^(\d+)(?:_|$)")

_STRUCTURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "brain": ("brain", "skull", "cranium", "ellipse"),
    "csp": ("csp", "cavum", "cavum_septi_pellucidi"),
    "lv": ("lv", "lateral_ventricle", "ventricle"),
}


class LargeScaleFetalHeadBiometryAdapter(BaseAdapter):
    DATASET_ID     = "Large-Scale-Fetal-Head-Biometry"
    ANATOMY_FAMILY = "fetal_head"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.3390/data8020038"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._pixel_size_maps = {
            group.group_dir: self._load_pixel_size_csv(group)
            for group in PLANE_GROUPS
        }

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Trans-thalamic").is_dir() and (root / "Diverse Fetal Head Images").is_dir():
            return root
        candidate = root / "large-scale-fetal-head-biometry"
        if (candidate / "Trans-thalamic").is_dir():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'Trans-thalamic/' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        samples = self._collect_samples()
        patient_ids = sorted({patient_id for _group, _path, patient_id in samples})
        split_map = {
            patient_id: self._infer_split(patient_id, idx, len(patient_ids))
            for idx, patient_id in enumerate(patient_ids)
        }

        mask_indices = {
            group.group_dir: self._build_mask_index(group, [
                img_path.stem
                for sample_group, img_path, _patient_id in samples
                if sample_group.group_dir == group.group_dir
            ])
            for group in PLANE_GROUPS
        }

        for group, img_path, patient_id in samples:
            stem = img_path.stem
            pixel_size_mm = self._pixel_size_maps.get(group.group_dir, {}).get(img_path.name)
            split = self.split_override or split_map.get(patient_id, "train")

            classification = self._make_instance(
                instance_id=f"{stem}_plane",
                label_raw=group.plane_label,
                label_ontology="fetal_head_plane",
                is_promptable=False,
                classification_label=group.plane_index,
            )

            mask_sources = mask_indices.get(group.group_dir, {}).get(stem, {})
            segmentation_instances = []
            materialized_masks: Dict[str, str] = {}
            for struct in STRUCTURES:
                mask_path = mask_sources.get(struct.key)
                if mask_path is None:
                    continue
                materialized_masks[struct.key] = str(mask_path)
                segmentation_instances.append(
                    self._make_instance(
                        instance_id=f"{stem}_{struct.key}",
                        label_raw=struct.label_raw,
                        label_ontology=struct.label_ontology,
                        mask_path=str(mask_path),
                        mask_channel=struct.mask_channel,
                        is_promptable=True,
                    )
                )

            has_mask = bool(segmentation_instances)
            instances = [classification] + segmentation_instances

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=patient_id,
                series_id=stem,
                view_type=self._view_type(group.plane_label),
                has_mask=has_mask,
                task_type="segmentation" if has_mask else "classification",
                ssl_stream="image",
                is_promptable=has_mask,
                source_meta={
                    "patient_id": patient_id,
                    "image_name": img_path.name,
                    "brain_plane": group.plane_label,
                    "source_subset": group.source_subset,
                    "image_set": "resized_959x661_rgb",
                    "original_size_used": False,
                    "pixel_size_mm": pixel_size_mm,
                    "annotation_source": "extracted_png_masks",
                    "annotation_geometries": {
                        struct.key: struct.geometry_type for struct in STRUCTURES
                    },
                    "mask_paths": materialized_masks,
                    "missing_mask_classes": [
                        struct.key for struct in STRUCTURES
                        if struct.key not in mask_sources
                    ],
                },
            )

    def _collect_samples(self) -> List[Tuple[PlaneGroup, Path, str]]:
        samples: List[Tuple[PlaneGroup, Path, str]] = []
        for group in PLANE_GROUPS:
            image_dir = self._image_dir(group)
            if not image_dir.exists():
                continue
            for img_path in sorted(image_dir.glob("*.png")):
                samples.append((group, img_path, self._patient_id(img_path.stem)))
        return samples

    def _load_pixel_size_csv(self, group: PlaneGroup) -> Dict[str, Optional[float]]:
        if group.pixel_csv is None:
            return {}
        csv_path = self._group_root(group) / group.pixel_csv
        if not csv_path.exists():
            return {}

        out: Dict[str, Optional[float]] = {}
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                label = normalized.get("Label") or normalized.get("label")
                px = self._first_value(
                    normalized,
                    "Pixel  in mm",
                    "Pixel in mm",
                    "pixel  in mm",
                    "pixel in mm",
                    "pixel_size_mm",
                )
                if label:
                    out[label] = self._to_float(px)
        return out

    def _build_mask_index(
        self,
        group: PlaneGroup,
        image_stems: Sequence[str],
    ) -> Dict[str, Dict[str, Path]]:
        stems = set(image_stems)
        index: Dict[str, Dict[str, Path]] = {}
        group_root = self._group_root(group)
        image_dir = self._image_dir(group)
        if not group_root.exists():
            return index

        # Extracted segmentation-mask PNGs.
        for png_path in sorted(group_root.rglob("*.png")):
            if image_dir in png_path.parents:
                continue
            stem, struct_key = self._parse_mask_identity(
                png_path.stem,
                png_path.parent.as_posix(),
                stems,
            )
            if stem and struct_key:
                index.setdefault(stem, {}).setdefault(
                    struct_key,
                    png_path,
                )

        return index

    @classmethod
    def _parse_mask_identity(
        cls,
        stem_candidate: str,
        context: str,
        image_stems: set[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        text = f"{stem_candidate} {context}".lower().replace("-", "_")
        struct_key = cls._structure_from_text(text)
        if struct_key is None:
            return None, None

        for stem in image_stems:
            normalized_stem = stem.lower().replace("-", "_")
            normalized_candidate = stem_candidate.lower().replace("-", "_")
            if normalized_candidate == normalized_stem:
                return stem, struct_key
            for alias in _STRUCTURE_ALIASES[struct_key]:
                for suffix in (f"_{alias}", f"__{alias}", f" {alias}"):
                    if normalized_candidate == f"{normalized_stem}{suffix}":
                        return stem, struct_key
        return None, None

    @staticmethod
    def _structure_from_text(text: str) -> Optional[str]:
        normalized = text.lower().replace("-", "_").replace("/", "_")
        for key, aliases in _STRUCTURE_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                return key
        return None

    def _group_root(self, group: PlaneGroup) -> Path:
        return self.root / group.group_dir

    def _image_dir(self, group: PlaneGroup) -> Path:
        return self._group_root(group) / group.image_subdir

    @staticmethod
    def _patient_id(stem: str) -> str:
        match = _PATIENT_RE.search(stem)
        if match:
            return match.group(1)
        match = _NUMERIC_PREFIX_RE.match(stem)
        if match:
            return match.group(1)
        return stem

    @staticmethod
    def _view_type(plane_label: str) -> str:
        return "fetal_head_" + plane_label.lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _first_value(row: dict, *keys: str) -> str:
        for key in keys:
            if key in row:
                return row[key]
        return ""

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        if value is None:
            return None
        value = str(value).strip().replace(",", ".")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
