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
3. stage_dataset() copies Store -> Scratch (rsync-friendly).
4. Root remapping via build_root_remap() so adapters write absolute Store
   paths and the dataloader transparently redirects to Scratch at runtime.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Anatomy -> dataset directory mapping ─────────────────────────────────────
# Maps dataset_id -> (anatomy_family, store_subdir)

DATASET_STORE_MAP: Dict[str, Tuple[str, str]] = {
    # Cardiac
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
    # Lung
    "COVIDx-US":           ("lung",            "COVIDx-US"),
    "LUS-multicenter-2025":("lung",            "LUS-multicenter-2025"),
    "POCUS-LUS":           ("lung",            "POCUS-LUS"),
    "COVID-BLUES":         ("lung",            "COVID-BLUES"),
    "ULTRASOUND-LUS":      ("lung",            "ULTRASOUND-LUS"),
    # Breast
    "BUS-BRA":             ("breast",          "BUS-BRA"),
    "BUSI":                ("breast",          "BUSI"),
    "BrEaST":              ("breast",          "BrEaST"),
    "BUS-UC":              ("breast",          "BUS-UC"),
    "BUS-UCLM":            ("breast",          "BUS-UCLM"),
    "BUID":                ("breast",          "BUID"),
    "STAnford-BUS":        ("breast",          "STAnford-BUS"),
    # Thyroid
    "TN3K":                ("thyroid",         "TN3K"),
    "TN5000":              ("thyroid",         "TN5000"),
    "TG3K":                ("thyroid",         "TG3K"),
    "TNSCUI":              ("thyroid",         "TNSCUI"),
    "DDTI":                ("thyroid",         "DDTI"),
    # Fetal
    "FETAL_PLANES_DB":     ("fetal",           "FETAL_PLANES_DB"),
    "HC18":                ("fetal",           "HC18"),
    "ACOUSLIC-AI":         ("fetal",           "ACOUSLIC-AI"),
    "OC4US":               ("fetal",           "OC4US"),
    # Kidney
    "KidneyUS":            ("kidney",          "KidneyUS"),
    # Liver
    "AUL":                 ("liver",           "AUL"),
    "BEHSOF":              ("liver",           "BEHSOF"),
    # Gallbladder
    "GBCU":                ("gallbladder",     "GBCU"),
    # Ovarian
    "MMOTU-2D":            ("ovarian",         "MMOTU-2D"),
    "PCOSGen":             ("ovarian",         "PCOSGen"),
    # Prostate
    "ProstateSeg":         ("prostate",        "ProstateSeg"),
    # Musculoskeletal
    "ASUS":                ("musculoskeletal", "ASUS"),
    "FALLMUD":             ("musculoskeletal", "FALLMUD"),
    "STMUS-NDA":           ("musculoskeletal", "STMUS-NDA"),
    # Multi
    "Unity-Imaging":       ("multi_organ",     "Unity-Imaging"),
    "STU-Hospital":        ("multi_organ",     "STU-Hospital"),
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
        elif user := os.environ.get("CSCS_USER"):
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

    def stage_dataset(
        self,
        dataset_id: str,
        dry_run: bool = False,
        rsync_args: str = "-av --progress",
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
        cmd = f"rsync {rsync_args} {src}/ {dst}/"
        log.info(f"Staging {dataset_id}: {src} -> {dst}")

        if dry_run:
            print(f"[DRY RUN] {cmd}")
            return True

        result = subprocess.run(cmd, shell=True)
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
