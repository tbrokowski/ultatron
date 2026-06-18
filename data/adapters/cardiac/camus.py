"""
data/adapters/cardiac/camus.py  ·  CAMUS dataset adapter
======================================================

CAMUS: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation
  500 patients, 2CH + 4CH views, ED + ES phases
  Labels: endocardium, epicardium (myocardium), left atrium
  Format: NIfTI .nii.gz files under database_nifti/

Actual layout after extracting the 'download' zip:
  {root}/database_nifti/patient{NNN}/
      patient{NNN}_2CH_ED.nii.gz
      patient{NNN}_2CH_ED_gt.nii.gz
      patient{NNN}_2CH_ES.nii.gz
      patient{NNN}_2CH_ES_gt.nii.gz
      patient{NNN}_2CH_half_sequence.nii.gz
      patient{NNN}_2CH_half_sequence_gt.nii.gz
      patient{NNN}_4CH_ED.nii.gz   ...
      Info_2CH.cfg                 (ED/ES frame indices, EF%, frame rate, …)
      Info_4CH.cfg

Entries emitted per patient × view:
  2 × image entries        (ED and ES single frames)
  1 × pseudo_video entry   (ED→ES pair, for short temporal SSL)
  1 × video entry          (half_sequence cine with per-frame masks)
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
from finetune.datasets.camus_io import official_split_for_patient, parse_patient_id


def _parse_info_cfg(cfg_path: Path) -> dict:
    """
    Parse CAMUS Info_{view}.cfg (KEY: VALUE lines).
    configparser lowercases keys, so ED → ed, NbFrame → nbframe, etc.
    """
    text = "[info]\n" + cfg_path.read_text()
    parser = configparser.ConfigParser()
    parser.read_string(text)
    sec = parser["info"]
    return {k: sec[k].strip() for k in sec}


def _view_meta(cfg_path: Path) -> dict:
    """Extract ED/ES indices, EF%, and acquisition metadata from an Info cfg."""
    meta: dict = {}
    if not cfg_path.exists():
        return meta
    try:
        raw = _parse_info_cfg(cfg_path)
        ed  = int(raw.get("ed", 0) or 0)
        es  = int(raw.get("es", 0) or 0)
        nb  = int(raw.get("nbframe", 0) or 0)
        ef  = float(raw.get("ef", 0) or 0)
        fps = float(raw.get("framerate", 0) or 0)
        meta.update({
            "ed_frame":      ed,
            "es_frame":      es,
            "nb_frames":     nb,
            "ef":            ef,
            "frame_rate":    fps,
            "age":           raw.get("age", ""),
            "sex":           raw.get("sex", ""),
            "image_quality": raw.get("imagequality", ""),
        })
    except (ValueError, TypeError):
        pass
    return meta


class CAMUSAdapter(BaseAdapter):
    """
    CAMUS adapter.

    Directory layout:
        {root}/database_nifti/patient{NNN}/
            patient{NNN}_2CH_ED.nii.gz   (image)
            patient{NNN}_2CH_ED_gt.nii.gz (segmentation mask)
            patient{NNN}_2CH_half_sequence.nii.gz  (cine volume)
            Info_2CH.cfg
            ...
    """

    DATASET_ID     = "CAMUS"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1109/TMI.2019.2900516"

    VIEWS   = ("2CH", "4CH")
    PHASES  = ("ED", "ES")

    def _patients_dir(self) -> Path:
        """Return the directory containing patient* subdirectories."""
        nifti_dir = self.root / "database_nifti"
        if nifti_dir.exists():
            return nifti_dir
        return self.root

    def _common_meta(self, pid: str, view: str, view_meta: dict) -> dict:
        """Shared source_meta fields for all CAMUS entries."""
        return {
            "patient_id": pid,
            "view":       view,
            **view_meta,
        }

    def iter_entries(self) -> Iterator[USManifestEntry]:
        pdir_root = self._patients_dir()
        patients  = sorted(pdir_root.glob("patient*/"))

        ext = ".nii.gz" if any(pdir_root.rglob("*.nii.gz")) else ".mhd"
        gt_suffix = f"_gt{ext}"

        for pdir in patients:
            pid   = pdir.name
            pid_num = parse_patient_id(pid)
            split = official_split_for_patient(pid_num) if pid_num else "train"
            if split is None:
                continue

            for view in self.VIEWS:
                cfg_path  = pdir / f"Info_{view}.cfg"
                view_meta = _view_meta(cfg_path)
                common    = self._common_meta(pid, view, view_meta)

                frames, masks = [], []
                for phase in self.PHASES:
                    img_path = pdir / f"{pid}_{view}_{phase}{ext}"
                    msk_path = pdir / f"{pid}_{view}_{phase}{gt_suffix}"
                    if not img_path.exists():
                        continue
                    frames.append(str(img_path))
                    masks.append(str(msk_path) if msk_path.exists() else None)

                if not frames:
                    continue

                has_mask = any(m is not None for m in masks)
                ef       = view_meta.get("ef")

                # ── Image entries (one per ED / ES frame) ─────────────────────
                for j, (fp, mp) in enumerate(zip(frames, masks)):
                    phase = self.PHASES[j]
                    instances = []
                    if mp:
                        instances.append(self._make_instance(
                            instance_id    = f"{pid}_{view}_{phase}",
                            label_raw      = "LV_myocardium",
                            label_ontology = "myocardium_lv",
                            mask_path      = mp,
                            is_promptable  = True,
                        ))
                    yield self._make_entry(
                        fp, split,
                        modality      = "image",
                        instances     = instances,
                        study_id      = pid,
                        view_type     = view,
                        has_mask      = bool(mp),
                        task_type     = "segmentation" if mp else "ssl_only",
                        ssl_stream    = "both",
                        is_promptable = bool(mp),
                        source_meta   = {**common, "phase": phase},
                    )

                # ── Pseudo-video entry (ED → ES pair) ─────────────────────────
                vid_instances = []
                for j, (fp, mp) in enumerate(zip(frames, masks)):
                    if mp:
                        vid_instances.append(self._make_instance(
                            instance_id    = f"{pid}_{view}_{self.PHASES[j]}_vid",
                            label_raw      = "LV_myocardium",
                            label_ontology = "myocardium_lv",
                            mask_path      = mp,
                            is_promptable  = True,
                        ))

                yield self._make_entry(
                    frames, split,
                    modality           = "pseudo_video",
                    instances          = vid_instances,
                    study_id           = pid,
                    view_type          = view,
                    num_frames         = len(frames),
                    is_cine            = True,
                    has_mask           = has_mask,
                    has_temporal_order = True,
                    task_type          = "segmentation" if has_mask else "ssl_only",
                    ssl_stream         = "both",
                    is_promptable      = has_mask,
                    source_meta        = common,
                )

                # ── Half-sequence cine (full clip + per-frame masks) ──────────
                half_seq = pdir / f"{pid}_{view}_half_sequence{ext}"
                half_gt  = pdir / f"{pid}_{view}_half_sequence{gt_suffix}"
                if not half_seq.exists():
                    continue

                nb_frames = view_meta.get("nb_frames") or 0
                has_half_mask = half_gt.exists()
                half_instances = []
                if has_half_mask:
                    half_instances.append(self._make_instance(
                        instance_id    = f"{pid}_{view}_half_seq",
                        label_raw      = "LV_myocardium",
                        label_ontology = "myocardium_lv",
                        mask_path      = str(half_gt),
                        is_promptable  = True,
                    ))

                # Volume-estimation labels (EF%) ride on cine entries; keep
                # segmentation task when masks exist so both heads can attach.
                half_task = "segmentation" if has_half_mask else (
                    "regression" if ef else "ssl_only"
                )

                yield self._make_entry(
                    str(half_seq), split,
                    modality           = "video",
                    instances          = half_instances,
                    study_id           = pid,
                    view_type          = view,
                    num_frames         = nb_frames,
                    is_cine            = True,
                    has_temporal_order = True,
                    has_mask           = has_half_mask,
                    task_type          = half_task,
                    ssl_stream         = "both",
                    is_promptable      = has_half_mask,
                    fps                = view_meta.get("frame_rate") or None,
                    source_meta        = common,
                )
