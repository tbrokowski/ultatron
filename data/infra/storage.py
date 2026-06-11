"""
storage.py  ·  CSCS capstor store & scratch path management
============================================================

All data paths are resolved through this module.
The storage layer is completely transparent to dataset adapters and
training code - they just call get_dataset_root() and receive a valid path.

Path hierarchy
--------------
Store  (permanent archive):
  /capstor/store/cscs/swissai/a127/ultrasound/
    raw/{anatomy_family}/{dataset_id}/
    manifests/
    checkpoints/

Scratch (fast I/O for training, ~30-day TTL):
  /capstor/scratch/cscs/{user}/ultrasound/
    raw/{anatomy_family}/{dataset_id}/
    manifests/
    cache/alp/        <- ALP saliency maps
    cache/frames/     <- pre-extracted video frames

Rules
-----
1. Store is the source of truth; never modify files there.
2. Training always reads from Scratch.
3. stage_dataset() copies Store -> Scratch (rsync when available, else cp -r).
4. Root remapping via build_root_remap() so adapters write absolute Store
   paths and the dataloader transparently redirects to Scratch at runtime.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Anatomy -> dataset directory mapping ─────────────────────────────────────
# Maps dataset_id -> (anatomy_family, store_subdir)

DATASET_STORE_MAP: Dict[str, Tuple[str, str]] = {
    # ── Cardiac ──────────────────────────────────────────────────────────────
    "CAMUS":                    ("cardiac", "CAMUS"),
    "EchoNet-Dynamic":          ("cardiac", "EchoNet-Dynamic"),
    "EchoNet-LVH":              ("cardiac", "EchoNet-LVH"),
    "EchoNet-Pediatric":        ("cardiac", "EchoNet-Pediatric"),
    "MIMIC-IV-ECHO":            ("cardiac", "MIMIC-IV-Echo"),
    "MIMIC-IV-Echo-LVVol-A4C":  ("cardiac", "MIMIC-IV-Echo-LVVol-A4C"),
    "MIMIC-EchoQA":             ("cardiac", "MIMIC-EchoQA"),
    "TED":                      ("cardiac", "TED"),
    "Unity-Echo":               ("cardiac", "Unity"),
    "CardiacUDC":               ("cardiac", "CardiacUDC"),
    "EchoCP":                   ("cardiac", "EchoCP"),
    "Echocardiogram-UCI":       ("cardiac", "Echocardiogram-UCI"),
    "CACTUS":                   ("cardiac", "CACTUS"),
    "MITEA":                    ("cardiac", "MITEA"),
    "PhysioNet-cardiac":        ("cardiac", "physionet.org"),
    # ── Lung ─────────────────────────────────────────────────────────────────
    "Benin-LUS":           ("lung", "Benin_Videos"),
    "RSA-LUS":             ("lung", "RSA_Videos"),
    "COVIDx-US":           ("lung", "COVIDx-US"),
    "LUS-multicenter-2025":("lung", "LUS-multicenter-2025"),
    "OpenPOCUS":           ("lung", "OpenPOCUS"),       # not yet in store
    "POCUS-LUS":           ("lung", "POCUS-LUS"),
    "COVID-BLUES":         ("lung", "COVID-BLUES"),
    "ULTRASOUND-LUS":      ("lung", "ULTRASOUND-LUS"),
    "LUSS-PHANTOM":        ("lung", "LUSS PHANTOM"),
    "Lung-Database":       ("lung", "Lung Database"),
    "Pocus-covid":         ("lung", "Pocus covid"),
    "BeninVideos":         ("lung", "BeninVideos"),
    "LUS-data":            ("lung", "data"),
    # ── Breast ───────────────────────────────────────────────────────────────
    "BUS-BRA":             ("breast", "BUSBRA"),
    "BUS-B":               ("breast", "Breast US B Dataset"),
    "BUSI":                ("breast", "BUSI"),
    "BrEaST":              ("breast", "BrEaST"),
    "BUS-UC":              ("breast", "BUS_UC"),
    "BUS-UCLM":            ("breast", "BUS-UCLM"),
    "BUID":                ("breast", "BUID"),
    "BUSV":                ("breast", "Miccai 2022 BUV Dataset"),
    "GDPH-SYSUCC":         ("breast", "GDPH&SYSUCC"),
    "Chinese-US-Report-Breast": ("breast", "Chinese US-Report Dataset (Breast)"),
    "S1":                  ("breast", "S1"),             # not yet in store
    "STAnford-BUS":        ("breast", "STAnford-BUS"),
    "busi-whu":            ("breast", "busi-whu"),
    "midi-b":              ("breast", "midi-b"),
    # ── Thyroid ──────────────────────────────────────────────────────────────
    "TN3K":                ("thyroid", "TN3K"),
    "TN5000":              ("thyroid", "TN5000"),
    "TG3K":                ("thyroid", "TG3K"),
    "TNSCUI":              ("thyroid", "TNSCUI"),
    "DDTI":                ("thyroid", "DDTI"),
    "Segthy-Dataset":      ("thyroid", "Segthy-Dataset"),
    "Thyroid-Nodule-Pathology": ("thyroid", "Thyroid_Nodule_Pathology"),
    "MuSeg":                    ("thyroid", "MuSeg"),
    "Micro-Ultrasound-Prostate-Segmentation": ("thyroid", "Micro_Ultrasound_Prostate_Segmentation"),
    # ── Fetal ────────────────────────────────────────────────────────────────
    "ACOUSLIC-AI":                         ("fetal", "ACOUSLIC"),
    "FASS":                                ("fetal", "fetal-abdominal-structures-segmentation"),
    "FETAL_PLANES_DB":                     ("fetal", "FETAL-PLANES-DB"),
    "FOCUS":                               ("fetal", "FOCUS"),
    "FPUS23":                              ("fetal", "FPUS23"),
    "FUGC":                                ("fetal", "FUGC"),
    "FH-PS-AOP":                           ("fetal", "FH-PS-AOP"),
    "HC18":                                ("fetal", "HC18"),
    "IUGC2024":                            ("fetal", "IUGC-2024"),
    "JNU-IFM":                             ("fetal", "JNU-IFM"),
    "Large-Scale-Fetal-Head-Biometry":     ("fetal", "large-scale-fetal-head-biometry"),
    "maternal-fetal-us-video-intrapartum": ("fetal", "maternal-fetal-us-video-intrapartum"),
    "OC4US":                               ("fetal", "OC4US"),
    "PBF-US1":                             ("fetal", "PBF-US1"),
    "PSFHS":                               ("fetal", "PSFHS"),
    "ultrasound-fetus-dataset":            ("fetal", "ultrasound-fetus-dataset"),
    "Fast-U-Net":                          ("fetal", "Fast-U-Net"),
    # ── Kidney ───────────────────────────────────────────────────────────────
    "KidneyUS":            ("kidney", "KidneyUS-US43d"),
    "Normal-Kidney-CV":    ("kidney", "Normal-Kidney-CV"),
    # ── Liver ────────────────────────────────────────────────────────────────
    "AUL":                 ("liver", "AUL"),
    "105US":               ("liver", "105US"),
    "fatty-liver-bmode":   ("liver", "fatty-liver-dataset"),
    "liver-CV-project":    ("liver", "liver-CV-project"),
    "LEPset":              ("liver", "LEPset-pancreas"),
    "BEHSOF":              ("liver", "BEHSOF"),
    "B-mode-CEUS-liver":   ("liver", "B-mode-CEUS-liver"),
    "C-TRUS":              ("liver", "C-TRUS"),
    "AbdomenUS-liver":     ("liver", "AbdomenUS/archive/abdominal_US/abdominal_US"),
    "ultrasound-elastography-liver-cancer": ("liver", "ultrasound-elastography-liver-cancer"),
    # ── Gallbladder / GI ─────────────────────────────────────────────────────
    "GBCU":                  ("gallbladder", "GBCU"),
    "GIST514-DB":            ("gallbladder", "GIST514-DB"),
    "RegensburgPedAppend":   ("gallbladder", "Regensburg Pediatric Appendicitis"),
    # ── Abdomen ──────────────────────────────────────────────────────────────
    "AbdomenUS":           ("abdominal", "abdominal_US"),
    "cptac-pda":           ("abdominal", "cptac-pda"),
    # ── Ovarian ──────────────────────────────────────────────────────────────
    "MMOTU-2D":            ("ovarian", "MMOTU-2D"),
    "PCOSGen":             ("ovarian", "PCOSGen"),
    "MMOTU-3D":            ("ovarian", "MMOTU-3D"),
    "Polycystic-Ovary-US-Telkom": ("ovarian", "Polycystic-Ovary-US-Telkom"),
    # ── Prostate ─────────────────────────────────────────────────────────────
    "ProstateSeg":             ("prostate", "openpros"),
    "Prostate-MRI-US-Biopsy":  ("prostate", "Prostate-MRI-US-Biopsy"),
    "muregpro":                ("prostate", "muregpro"),
    # ── Musculoskeletal ──────────────────────────────────────────────────────
    "FALLMUD":             ("musculoskeletal", "FALLMUD"),
    "LUMINOUS":            ("musculoskeletal", "LUMINOUS_Database"),
    "deepMTJ":             ("musculoskeletal", "Muscle-Tendon Junction Tracking"),
    "KneeUSJoCoHS":        ("musculoskeletal", "knee us dataset"),
    "STMUS-NDA":           ("musculoskeletal", "STMUS NDA "),
    "TUS-REC":             ("musculoskeletal", "TUS-REC (Freehand 3D US Arm:Forearm)"),
    "TUS-REC-Val":         ("musculoskeletal", "Reconstructing 2D to 3D US (Forearms)"),
    "SpinalCordInjuryUS":  ("musculoskeletal", "Spinal Cord Injury US (Sci. Reports 2025)"),
    "msk-heckmatt-radboud": ("musculoskeletal", "msk-heckmatt-radboud"),
    "msk-nmd-radboud":      ("musculoskeletal", "msk-nmd-radboud"),
    "open-hip-dysplasia":   ("musculoskeletal", "open-hip-dysplasia"),
    # ── Nerve ────────────────────────────────────────────────────────────────
    "optic-nerve-sheaths":  ("nerve", "optic-nerve-sheaths"),
    "us-guided-anesthesia": ("nerve", "us guided anesthesia"),
    # ── Vascular / carotid ───────────────────────────────────────────────────
    "CUBS":                                ("vascular-carotid", "CUBS"),
    "Common-Carotid-Artery-Ultrasound-Images": ("vascular-carotid", "Common-Carotid-Artery-Ultrasound-Images"),
    # ── Brain ────────────────────────────────────────────────────────────────
    "3D-US-Neuroimages-Dataset": ("brain", "3D-US-Neuroimages-Dataset"),
    "BITE":                      ("brain", "BITE"),
    "REMIND-Brain-iUS":          ("brain", "REMIND-Brain-iUS"),
    "RESECT":                    ("brain", "RESECT"),
    "ReMIND2Reg":                ("brain", "ReMIND2Reg"),
    "braTioUS":                  ("brain", "braTioUS"),
    # ── Multi-organ ──────────────────────────────────────────────────────────
    "STU-Hospital-master":           ("multi_organ", "STU-Hospital-master"),
    "annotated_heterogeneous_us_db": ("multi_organ", "annotated_heterogeneous_us_db"),
    "US-365K":                       ("multi_organ", "US-365K"),
    # ── Ocular ───────────────────────────────────────────────────────────────
    "ERDES":                     ("ocular", "ERDES"),
    # ── Skin ─────────────────────────────────────────────────────────────────
    "Dermatologic-US-Skin-Lesions": ("skin", "Dermatologic-US-Skin-Lesions"),
}


@dataclass
class StorageConfig:
    """
    Runtime storage configuration.

    Environment variable overrides:
      US_STORE_ROOT   : overrides store_root
      US_SCRATCH_ROOT : overrides scratch_root
      CSCS_USER       : username for scratch path template
      US_LOCAL_DEV_ROOT: local dev fallback
    """
    store_root: Path = Path("/capstor/store/cscs/swissai/a127/ultrasound")
    scratch_root: Optional[Path] = None
    use_scratch: bool = True
    local_dev_root: Optional[Path] = None

    def __post_init__(self):
        if env := os.environ.get("US_STORE_ROOT"):
            self.store_root = Path(env)
        if env := os.environ.get("US_SCRATCH_ROOT"):
            self.scratch_root = Path(env)
        elif self.scratch_root is None:
            user = (
                os.environ.get("CSCS_USER")
                or os.environ.get("CSCS_USERNAME")
                or os.environ.get("USER")
            )
            if user:
                self.scratch_root = Path(f"/capstor/scratch/cscs/{user}/ultrasound")
        if env := os.environ.get("US_LOCAL_DEV_ROOT"):
            self.local_dev_root = Path(env)

    @property
    def active_root(self) -> Path:
        if self.use_scratch and self.scratch_root and self.scratch_root.exists():
            return self.scratch_root
        if self.store_root.exists():
            return self.store_root
        if self.local_dev_root and self.local_dev_root.exists():
            log.warning(f"Falling back to local dev root: {self.local_dev_root}")
            return self.local_dev_root
        raise RuntimeError(
            f"No valid data root. store={self.store_root}, "
            f"scratch={self.scratch_root}, local_dev={self.local_dev_root}"
        )

    def raw_root(self, use_scratch: Optional[bool] = None) -> Path:
        use = use_scratch if use_scratch is not None else self.use_scratch
        if use and self.scratch_root and self.scratch_root.exists():
            return self.scratch_root / "raw"
        return self.store_root / "raw"

    def manifests_root(self, use_scratch: Optional[bool] = None) -> Path:
        use = use_scratch if use_scratch is not None else self.use_scratch
        if use and self.scratch_root and self.scratch_root.exists():
            return self.scratch_root / "manifests"
        return self.store_root / "manifests"

    def alp_cache_root(self) -> Path:
        base = self.scratch_root or self.store_root
        return base / "cache" / "alp"

    def frames_cache_root(self) -> Path:
        base = self.scratch_root or self.store_root
        return base / "cache" / "frames"

    def get_dataset_root(
        self, dataset_id: str, anatomy_family: Optional[str] = None
    ) -> Path:
        if dataset_id in DATASET_STORE_MAP:
            anatomy, subdir = DATASET_STORE_MAP[dataset_id]
        else:
            anatomy = anatomy_family or "other"
            subdir = dataset_id
        return self.raw_root() / anatomy / subdir


    # ── Compatibility helpers ────────────────────────────────────────────────

    @property
    def user(self) -> str:
        """CSCS username extracted from scratch_root."""
        if self.scratch_root:
            parts = str(self.scratch_root).split("/")
            try:
                cscs_idx = parts.index("cscs")
                return parts[cscs_idx + 1]
            except (ValueError, IndexError):
                pass
        return os.environ.get("USER", "unknown")

    def manifest_path(self, split: str = "train", scratch: bool = True) -> "Path":
        base = self.manifests_root(use_scratch=scratch)
        return base / f"us_foundation_{split}.jsonl"

    def alp_cache_path(self, sample_id: str) -> "Path":
        bucket = sample_id[:2].upper()
        return self.alp_cache_root() / bucket / f"{sample_id}.pt"

    def root_remap(self, source: str = "scratch") -> "Dict[str, str]":
        remap = self.build_root_remap()
        if source == "scratch":
            return remap
        elif source == "store":
            return {v: k for k, v in remap.items()}
        return {}

    def build_root_remap(self) -> Dict[str, str]:
        """
        Build {store_path: scratch_path} dict for manifest path remapping.
        Pass to USFoundationDataset.root_remap at runtime.
        """
        if not self.use_scratch or not self.scratch_root:
            return {}
        return {
            str(self.store_root / "raw"): str(self.scratch_root / "raw"),
        }

    @staticmethod
    def _has_rsync() -> bool:
        return shutil.which("rsync") is not None

    def _build_stage_cmd(
        self,
        src: Path,
        dst: Path,
        rsync_args: str = "-ah --info=progress2",
    ) -> list[str]:
        """Build a copy command that safely handles spaces/parentheses in paths."""
        if self._has_rsync():
            return ["rsync", *rsync_args.split(), f"{src}/", f"{dst}/"]
        return ["cp", "-r", f"{src}/.", f"{dst}/"]

    def stage_dataset(
        self,
        dataset_id: str,
        dry_run: bool = False,
        rsync_args: str = "-ah --info=progress2",
    ) -> bool:
        if not self.scratch_root:
            log.error("scratch_root not configured. Cannot stage.")
            return False

        anatomy, subdir = DATASET_STORE_MAP.get(dataset_id, ("other", dataset_id))
        src = self.store_root / "raw" / anatomy / subdir
        dst = self.scratch_root / "raw" / anatomy / subdir

        if not src.exists():
            log.error(f"Source not found: {src}")
            return False

        dst.mkdir(parents=True, exist_ok=True)
        cmd = self._build_stage_cmd(src, dst, rsync_args=rsync_args)
        copy_tool = "rsync" if self._has_rsync() else "cp"
        log.info(f"Staging {dataset_id} ({copy_tool}): {src} -> {dst}")

        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            return True

        result = subprocess.run(cmd)
        success = result.returncode == 0
        if not success:
            log.error(f"Failed to stage {dataset_id} (code {result.returncode})")
        return success

    def stage_all(
        self,
        dataset_ids: Optional[List[str]] = None,
        anatomy_family: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, bool]:
        if dataset_ids is None:
            if anatomy_family:
                dataset_ids = [
                    did for did, (af, _) in DATASET_STORE_MAP.items()
                    if af == anatomy_family
                ]
            else:
                dataset_ids = list(DATASET_STORE_MAP.keys())
        return {did: self.stage_dataset(did, dry_run=dry_run) for did in dataset_ids}

    def dataset_is_staged(self, dataset_id: str) -> bool:
        if not self.scratch_root:
            return False
        anatomy, subdir = DATASET_STORE_MAP.get(dataset_id, ("other", dataset_id))
        dst = self.scratch_root / "raw" / anatomy / subdir
        return dst.exists() and any(dst.iterdir())

    def dataset_is_in_store(self, dataset_id: str) -> bool:
        anatomy, subdir = DATASET_STORE_MAP.get(dataset_id, ("other", dataset_id))
        src = self.store_root / "raw" / anatomy / subdir
        return src.exists() and any(src.iterdir())

    def resolve_dataset_root(self, dataset_id: str, prefer_scratch: bool = True) -> Optional[Path]:
        """Return scratch or store path for a dataset if it exists and is non-empty."""
        anatomy, subdir = DATASET_STORE_MAP.get(dataset_id, ("other", dataset_id))
        candidates: List[Path] = []
        if prefer_scratch and self.scratch_root:
            candidates.append(self.scratch_root / "raw" / anatomy / subdir)
        candidates.append(self.store_root / "raw" / anatomy / subdir)
        for path in candidates:
            if path.exists() and any(path.iterdir()):
                return path
        return None

    def resolve_all_dataset_roots(self, prefer_scratch: bool = True) -> Dict[str, str]:
        """Map every registered dataset_id to an existing root path (scratch preferred)."""
        roots: Dict[str, str] = {}
        for dataset_id in sorted(DATASET_STORE_MAP.keys()):
            path = self.resolve_dataset_root(dataset_id, prefer_scratch=prefer_scratch)
            if path is not None:
                roots[dataset_id] = str(path)
        return roots

    def status_report(self) -> str:
        lines = [
            f"{'Dataset':<30} {'Anatomy':<20} {'Store':<8} {'Scratch':<8}",
            "-" * 68,
        ]
        for did, (af, _) in sorted(DATASET_STORE_MAP.items(), key=lambda x: (x[1][0], x[0])):
            s = "OK" if self.dataset_is_in_store(did) else "--"
            c = "OK" if self.dataset_is_staged(did) else "--"
            lines.append(f"{did:<30} {af:<20} {s:<8} {c:<8}")
        return "\n".join(lines)


# ── Default singleton ─────────────────────────────────────────────────────────

_DEFAULT_STORAGE: Optional[StorageConfig] = None


def get_storage_config(user: Optional[str] = None) -> StorageConfig:
    """Alias for backward compatibility."""
    return get_storage(user)


def get_storage(user: Optional[str] = None) -> StorageConfig:
    global _DEFAULT_STORAGE
    if _DEFAULT_STORAGE is None:
        _DEFAULT_STORAGE = StorageConfig()
    return _DEFAULT_STORAGE


def configure_storage(
    store_root: Optional[str] = None,
    scratch_root: Optional[str] = None,
    use_scratch: bool = True,
    local_dev_root: Optional[str] = None,
) -> StorageConfig:
    """Configure the global storage singleton. Call once at startup."""
    global _DEFAULT_STORAGE
    base = StorageConfig()
    _DEFAULT_STORAGE = StorageConfig(
        store_root=Path(store_root) if store_root else base.store_root,
        scratch_root=Path(scratch_root) if scratch_root else None,
        use_scratch=use_scratch,
        local_dev_root=Path(local_dev_root) if local_dev_root else None,
    )
    return _DEFAULT_STORAGE
