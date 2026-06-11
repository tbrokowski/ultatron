"""
data/adapters/cardiac/cactus.py  ·  CACTUS cardiac ultrasound adapter
======================================================================

Cardiac Assessment and Classification of Ultrasound (CACTUS) — phantom
echocardiography images with expert view labels and quality grades (0–10).

On-disk layout (FRDR / Academic Torrents):
  {root}/Cactus Dataset/
    Images Dataset/{A4C,SC,PL,PSAV,PSMV,Random}/*.jpg
    Grades/{view}_grades.csv     Image Name, Subfolder Name, Grade
    Videos/
      Training/Video{1..5}.mp4 + Video{1..5}.csv
      Real Time Scan/*.mp4

Yields one image entry per graded frame (37,736) plus training/demo videos.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_VIEW_META = {
    "A4C":    {"view_type": "A4C",         "class_idx": 0, "ontology": "cardiac_view_a4c"},
    "SC":     {"view_type": "SC",          "class_idx": 1, "ontology": "cardiac_view_sc"},
    "PL":     {"view_type": "PLAX",        "class_idx": 2, "ontology": "cardiac_view_plax"},
    "PSAV":   {"view_type": "PSAX",        "class_idx": 3, "ontology": "cardiac_view_psax"},
    "PSMV":   {"view_type": "PSAX",        "class_idx": 4, "ontology": "cardiac_view_psax"},
    "Random": {"view_type": "non_cardiac", "class_idx": 5, "ontology": "cardiac_view_random"},
}

_VIDEO_EXTS = {".mp4", ".avi", ".mov"}


class CACTUSAdapter(BaseAdapter):
    DATASET_ID = "CACTUS"
    ANATOMY_FAMILY = "cardiac"
    SONODQS = "silver"
    DOI = "https://doi.org/10.20383/103.01484"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self.data_root = self.root
        self.images_root = self.data_root / "Images Dataset"
        self.grades_dir = self.data_root / "Grades"
        self.videos_root = self.data_root / "Videos"

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        for candidate in (root, root / "Cactus Dataset"):
            if (candidate / "Images Dataset").is_dir() and (candidate / "Grades").is_dir():
                return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'Cactus Dataset/Images Dataset' and "
            f"'Cactus Dataset/Grades' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        image_records = self._load_image_records()
        n = len(image_records)
        for i, rec in enumerate(image_records):
            split = self.split_override or self._infer_split(rec["image_name"], i, n)
            view = rec["view"]
            meta = _VIEW_META[view]
            grade = rec["grade"]

            instance = self._make_instance(
                instance_id=f"{rec['image_name']}_{view}",
                label_raw=view,
                label_ontology=meta["ontology"],
                is_promptable=False,
                classification_label=meta["class_idx"],
            )

            yield self._make_entry(
                rec["image_path"],
                split,
                modality="image",
                instances=[instance],
                study_id=rec["image_name"],
                view_type=meta["view_type"],
                task_type="classification",
                ssl_stream="image",
                is_promptable=False,
                has_mask=False,
                source_meta={
                    "view": view,
                    "quality_grade": grade,
                    "subfolder": rec["subfolder"],
                },
            )

        yield from self._iter_training_videos()
        yield from self._iter_demo_videos()

    def _load_image_records(self) -> list[dict]:
        records: list[dict] = []
        for csv_path in sorted(self.grades_dir.glob("*_grades.csv")):
            view = csv_path.stem.replace("_grades", "")
            if view not in _VIEW_META:
                log.warning("CACTUS: skipping unknown grades file %s", csv_path.name)
                continue

            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    image_name = (row.get("Image Name") or row.get("image_name") or "").strip()
                    subfolder = (row.get("Subfolder Name") or row.get("subfolder") or view).strip()
                    grade_raw = (row.get("Grade") or row.get("grade") or "").strip()
                    if not image_name:
                        continue

                    img_path = self.images_root / subfolder / image_name
                    if not img_path.is_file():
                        alt = self.images_root / view / image_name
                        if alt.is_file():
                            img_path = alt
                        else:
                            continue

                    try:
                        grade = int(float(grade_raw))
                    except (TypeError, ValueError):
                        continue

                    records.append(
                        {
                            "image_name": image_name,
                            "image_path": str(img_path),
                            "view": view,
                            "subfolder": subfolder,
                            "grade": grade,
                        }
                    )

        records.sort(key=lambda r: (r["view"], r["image_name"]))
        return records

    def _iter_training_videos(self) -> Iterator[USManifestEntry]:
        training_dir = self.videos_root / "Training"
        if not training_dir.is_dir():
            return

        videos = sorted(
            p for p in training_dir.iterdir()
            if p.suffix.lower() in _VIDEO_EXTS
        )
        n = len(videos)
        for i, vpath in enumerate(videos):
            csv_path = vpath.with_suffix(".csv")
            frame_labels = self._load_video_csv(csv_path) if csv_path.is_file() else []
            split = self.split_override or self._infer_split(vpath.stem, i, n)

            yield self._make_entry(
                str(vpath),
                split,
                modality="video",
                instances=[],
                study_id=vpath.stem,
                view_type="mixed",
                is_cine=True,
                has_temporal_order=True,
                task_type="ssl_only",
                ssl_stream="video",
                is_promptable=False,
                has_mask=False,
                source_meta={
                    "video_role": "training",
                    "frame_labels": frame_labels,
                    "n_labeled_frames": len(frame_labels),
                },
            )

    def _iter_demo_videos(self) -> Iterator[USManifestEntry]:
        demo_dir = self.videos_root / "Real Time Scan"
        if not demo_dir.is_dir():
            return

        videos = sorted(
            p for p in demo_dir.iterdir()
            if p.suffix.lower() in _VIDEO_EXTS
        )
        for i, vpath in enumerate(videos):
            split = self.split_override or "test"
            yield self._make_entry(
                str(vpath),
                split,
                modality="video",
                instances=[],
                study_id=vpath.stem,
                view_type="mixed",
                is_cine=True,
                has_temporal_order=True,
                task_type="ssl_only",
                ssl_stream="video",
                is_promptable=False,
                has_mask=False,
                source_meta={
                    "video_role": "real_time_demo",
                },
            )

    @staticmethod
    def _load_video_csv(csv_path: Path) -> list[dict]:
        labels: list[dict] = []
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return labels

            fields = {name.strip().lower(): name for name in reader.fieldnames if name}
            for row in reader:
                frame = (row.get(fields.get("frame", "frame")) or "").strip()
                if not frame:
                    continue

                rec: dict = {"frame": frame}
                if "grade" in fields:
                    rec["grade"] = row.get(fields["grade"])
                if "score" in fields:
                    rec["score"] = row.get(fields["score"])
                if "view" in fields:
                    rec["view"] = row.get(fields["view"])
                if "name" in fields:
                    rec["name"] = row.get(fields["name"])
                labels.append(rec)
        return labels
