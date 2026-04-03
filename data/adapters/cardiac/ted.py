"""
data/adapters/cardiac/ted.py  ·  TED (Temporal Echocardiography Dataset) adapter
==================================================================================

TED: 100 patients, 4-chamber view cine sequences.
  Labels:  per-frame LV segmentation masks, EF, ED/ES frame indices
  Format:  SimpleITK .mhd + .raw files (3D: H × W × T)
  Source:  https://www.creatis.insa-lyon.fr/Challenge/ted/

Entries emitted per patient:
  1 × video entry   — full cine sequence (.mhd volume), LV mask
  2 × image entries — individual ED and ES frames (frame_idx stored in source_meta
                      so the dataloader can slice the volume at read time)
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


def _parse_cfg(cfg_path: Path) -> dict:
    """
    Parse the .cfg info file which uses a simple KEY: VALUE format.
    configparser requires a section header, so we prefix one.
    """
    text = "[info]\n" + cfg_path.read_text()
    parser = configparser.ConfigParser()
    parser.read_string(text)
    sec = parser["info"]
    return {k: sec[k].strip() for k in sec}


class TEDAdapter(BaseAdapter):
    """
    TED adapter.

    Directory layout:
        {root}/database/
            patient{NNN}/
                patient{NNN}_4CH_sequence.mhd      # 3-D cine volume (H×W×T)
                patient{NNN}_4CH_sequence_gt.mhd   # per-frame segmentation mask
                patient{NNN}_4CH_sequence.raw       # SimpleITK binary data
                patient{NNN}_4CH_sequence_gt.raw
                patient{NNN}_4CH_info.cfg           # ED, ES, NbFrame, EF, …
    """

    DATASET_ID     = "TED"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://www.creatis.insa-lyon.fr/Challenge/ted/"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        db_dir   = self.root / "database"
        patients = sorted(db_dir.glob("patient*/")) if db_dir.exists() else []
        n        = len(patients)

        for i, pdir in enumerate(patients):
            pid  = pdir.name
            seq  = pdir / f"{pid}_4CH_sequence.mhd"
            mask = pdir / f"{pid}_4CH_sequence_gt.mhd"
            cfg  = pdir / f"{pid}_4CH_info.cfg"

            if not seq.exists():
                continue

            split = self._infer_split(pid, i, n)

            meta: dict = {}
            ed_frame = es_frame = nb_frames = None
            ef = None
            if cfg.exists():
                try:
                    raw_meta = _parse_cfg(cfg)
                    ed_frame  = int(raw_meta.get("ed", 1))
                    es_frame  = int(raw_meta.get("es", 1))
                    nb_frames = int(raw_meta.get("nbframe", 0))
                    ef        = float(raw_meta.get("ef", 0.0))
                    meta.update({
                        "ed_frame":  ed_frame,
                        "es_frame":  es_frame,
                        "nb_frames": nb_frames,
                        "ef":        ef,
                        "age":       raw_meta.get("age", ""),
                        "sex":       raw_meta.get("sex", ""),
                        "image_quality": raw_meta.get("imagequality", ""),
                    })
                except Exception:
                    pass

            has_mask    = mask.exists()
            mask_str    = str(mask) if has_mask else None

            # ── Full-sequence video entry ─────────────────────────────────────
            vid_instances = []
            if has_mask:
                vid_instances.append(self._make_instance(
                    instance_id    = f"{pid}_4CH_seq",
                    label_raw      = "LV_myocardium",
                    label_ontology = "myocardium_lv",
                    mask_path      = mask_str,
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(seq), split,
                modality           = "video",
                instances          = vid_instances,
                study_id           = pid,
                view_type          = "4CH",
                is_cine            = True,
                has_temporal_order = True,
                num_frames         = nb_frames or 0,
                has_mask           = has_mask,
                task_type          = "segmentation" if has_mask else "ssl_only",
                ssl_stream         = "both",
                is_promptable      = has_mask,
                source_meta        = {"root": str(self.root), "doi": self.DOI, **meta},
            )

            # ── ED / ES single-frame image entries ────────────────────────────
            # The .mhd volume must be sliced at runtime using frame_idx.
            for phase, frame_idx in (("ED", ed_frame), ("ES", es_frame)):
                if frame_idx is None:
                    continue
                img_instances = []
                if has_mask:
                    img_instances.append(self._make_instance(
                        instance_id    = f"{pid}_4CH_{phase}",
                        label_raw      = "LV_myocardium",
                        label_ontology = "myocardium_lv",
                        mask_path      = mask_str,
                        is_promptable  = True,
                    ))
                yield self._make_entry(
                    str(seq), split,
                    modality      = "image",
                    instances     = img_instances,
                    study_id      = pid,
                    view_type     = "4CH",
                    has_mask      = has_mask,
                    task_type     = "segmentation" if has_mask else "ssl_only",
                    ssl_stream    = "both",
                    is_promptable = has_mask,
                    source_meta   = {
                        "root":      str(self.root),
                        "doi":       self.DOI,
                        "phase":     phase,
                        "frame_idx": frame_idx,
                        **meta,
                    },
                )
