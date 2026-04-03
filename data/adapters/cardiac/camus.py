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
      patient{NNN}_4CH_ED.nii.gz   ...

Entries emitted per patient × view × phase:
  1 × image entry       (single ED or ES frame)
  1 × pseudo_video entry (ED→ES pair, for video branch)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class CAMUSAdapter(BaseAdapter):
    """
    CAMUS adapter.

    Directory layout:
        {root}/patient{NNN}/
            patient{NNN}_2CH_ED.mhd   (image)
            patient{NNN}_2CH_ED_gt.mhd (segmentation mask)
            patient{NNN}_2CH_ED.zraw   (SimpleITK data file)
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
        # Downloaded as: {root}/database_nifti/patient*/
        nifti_dir = self.root / "database_nifti"
        if nifti_dir.exists():
            return nifti_dir
        # Fallback: root itself (legacy .mhd layout)
        return self.root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        pdir_root = self._patients_dir()
        patients  = sorted(pdir_root.glob("patient*/"))
        n         = len(patients)

        # Support both .nii.gz (downloaded NIfTI) and .mhd (legacy)
        ext = ".nii.gz" if any(pdir_root.rglob("*.nii.gz")) else ".mhd"
        gt_suffix = f"_gt{ext}"

        for i, pdir in enumerate(patients):
            pid   = pdir.name
            split = self._infer_split(pid, i, n)

            for view in self.VIEWS:
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

                # ── Image entries (one per frame) ─────────────────────────────
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
                        modality   = "image",
                        instances  = instances,
                        study_id   = pid,
                        view_type  = view,
                        has_mask   = bool(mp),
                        task_type  = "segmentation" if mp else "ssl_only",
                        ssl_stream = "both",
                        is_promptable = bool(mp),
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
                    modality          = "pseudo_video",
                    instances         = vid_instances,
                    study_id          = pid,
                    view_type         = view,
                    num_frames        = len(frames),
                    is_cine           = True,
                    has_mask          = has_mask,
                    has_temporal_order= True,
                    task_type         = "segmentation" if has_mask else "ssl_only",
                    ssl_stream        = "video",
                    is_promptable     = has_mask,
                )
