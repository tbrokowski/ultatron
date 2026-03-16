"""
cscs_paths.py  ·  CSCS Store / Scratch path management
=======================================================

Manages the two-tier storage layout on CSCS Alps / Capstor:

  Store  (permanent):  /capstor/store/cscs/swissai/a127/ultrasound/
  Scratch (fast I/O):  /capstor/scratch/cscs/$USER/ultrasound/

Key capabilities
----------------
* Automatic path resolution — prefers Scratch over Store for training
* remap_dict() for USFoundationDataModule.root_remap
* generate_stage_script() — produces a ready-to-run bash staging script
* Validation helpers — check which datasets are staged and quota
* Works transparently in local dev (paths don't exist → graceful fallback)

Usage
-----
  # On CSCS
  cfg = CSCSConfig.from_env()
  dm = USFoundationDataModule(
      manifest_path=str(cfg.manifest_path("us_foundation_train.jsonl")),
      root_remap=cfg.remap_dict(),
  )

  # Local dev with custom root
  cfg = CSCSConfig(
      store_root=Path("/data/ultrasound"),
      scratch_root=Path("/data/ultrasound"),  # same dir locally
  )
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Fixed CSCS paths ──────────────────────────────────────────────────────────

CSCS_STORE_ROOT  = "/capstor/store/cscs/swissai/a127/ultrasound"
CSCS_SCRATCH_TPL = "/capstor/scratch/cscs/{user}/ultrasound"

# Anatomy families → store subdirectory
ANATOMY_STORE_DIRS = [
    "cardiac", "lung", "breast", "thyroid", "fetal", "kidney",
    "liver", "ovarian", "prostate", "musculoskeletal", "multi_organ",
]


# ══════════════════════════════════════════════════════════════════════════════
# CSCSConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CSCSConfig:
    """
    Encapsulates all path logic for the two-tier CSCS storage.

    Attributes
    ----------
    store_root   : Path to permanent archive (never purged).
    scratch_root : Path to fast Lustre scratch (purged after ~30 days inactivity).
    prefer_scratch: If True, resolve dataset paths from scratch first.
    """
    store_root:    Path = field(default_factory=lambda: Path(CSCS_STORE_ROOT))
    scratch_root:  Path = field(default_factory=lambda: Path("/tmp/ultrasound"))
    prefer_scratch: bool = True

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "CSCSConfig":
        """Build from environment variable $USER (standard on CSCS login nodes)."""
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
        return cls.from_user(user)

    @classmethod
    def from_user(cls, username: str) -> "CSCSConfig":
        scratch = CSCS_SCRATCH_TPL.format(user=username)
        return cls(
            store_root=Path(CSCS_STORE_ROOT),
            scratch_root=Path(scratch),
        )

    @classmethod
    def local_dev(cls, base_dir: str | Path) -> "CSCSConfig":
        """For local development when CSCS paths don't exist."""
        base = Path(base_dir)
        return cls(store_root=base, scratch_root=base, prefer_scratch=False)

    @classmethod
    def from_yaml(cls, cfg: dict) -> "CSCSConfig":
        """Build from a data_config.yaml dict."""
        storage = cfg.get("storage", {})
        user = storage.get("user") or os.environ.get("USER", "unknown")
        store = storage.get("store_root", CSCS_STORE_ROOT)
        scratch = storage.get("scratch_root", CSCS_SCRATCH_TPL.format(user=user))
        return cls(
            store_root=Path(store),
            scratch_root=Path(scratch),
            prefer_scratch=storage.get("prefer_scratch", True),
        )

    # ── Core path builders ────────────────────────────────────────────────────

    def store_path(self, *parts: str) -> Path:
        return self.store_root.joinpath(*parts)

    def scratch_path(self, *parts: str) -> Path:
        return self.scratch_root.joinpath(*parts)

    def dataset_root(
        self,
        anatomy: str,
        dataset_id: str,
    ) -> Path:
        """
        Resolve a dataset root path.

        Prefers Scratch (fast) if staged; falls back to Store.
        Returns the best available path even if it does not exist yet.
        """
        scratch = self.scratch_root / "raw" / anatomy / dataset_id
        store   = self.store_root   / "raw" / anatomy / dataset_id

        if self.prefer_scratch and scratch.exists():
            return scratch
        if store.exists():
            return store
        # Neither exists — return Scratch as the canonical target for staging
        return scratch

    def manifest_path(self, name: str) -> Path:
        """
        Resolve a manifest path.

        Prefers Scratch; falls back to Store.
        """
        scratch = self.scratch_root / "manifests" / name
        store   = self.store_root   / "manifests" / name
        if self.prefer_scratch and scratch.exists():
            return scratch
        if store.exists():
            return store
        return scratch  # default write target

    def alp_cache_dir(self) -> Path:
        return self.scratch_root / "alp_cache"

    def checkpoints_dir(self, phase: int = 1) -> Path:
        return self.scratch_root / "checkpoints" / f"phase{phase}"

    def store_checkpoints_dir(self, phase: int = 1) -> Path:
        return self.store_root / "checkpoints" / f"phase{phase}"

    # ── DataModule integration ────────────────────────────────────────────────

    def remap_dict(self) -> Dict[str, str]:
        """
        Returns root_remap dict for USFoundationDataModule.

        When manifests were built against Store paths, this remaps them
        transparently to Scratch paths at load time.
        """
        return {str(self.store_root): str(self.scratch_root)}

    def dataset_roots_for_config(
        self, anatomy_map: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Build a {dataset_id: resolved_path} dict for data_config.yaml's
        'datasets' section. anatomy_map: {dataset_id: anatomy_family}.
        """
        return {
            ds_id: str(self.dataset_root(anatomy, ds_id))
            for ds_id, anatomy in anatomy_map.items()
        }

    # ── Staging helpers ────────────────────────────────────────────────────────

    def is_staged(self, anatomy: str, dataset_id: str) -> bool:
        """Check whether a dataset has been copied to Scratch."""
        target = self.scratch_root / "raw" / anatomy / dataset_id
        return target.exists() and any(target.iterdir())

    def staged_datasets(
        self, anatomy_map: Dict[str, str]
    ) -> Dict[str, bool]:
        """
        Returns {dataset_id: is_staged} for all datasets in anatomy_map.

        anatomy_map: {dataset_id: anatomy_family}
        """
        return {
            ds: self.is_staged(anatomy, ds)
            for ds, anatomy in anatomy_map.items()
        }

    def generate_stage_script(
        self,
        datasets: List[Tuple[str, str]],  # [(anatomy, dataset_id), ...]
        lustre_stripe_count: int = 4,
        video_datasets: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a ready-to-run bash staging script.

        datasets         : list of (anatomy_folder, dataset_id) tuples
        video_datasets   : dataset IDs that need Lustre striping (for large video files)
        """
        video_datasets = set(video_datasets or ["EchoNet-Dynamic", "MIMIC-IV-ECHO"])

        lines = [
            "#!/bin/bash",
            "# stage_datasets.sh — auto-generated by CSCSConfig.generate_stage_script()",
            "# Run this before submitting a training job.",
            "",
            f'STORE="{self.store_root}"',
            f'SCRATCH="{self.scratch_root}"',
            "",
            "set -euo pipefail",
            "",
            "# ── Pre-create directories ────────────────────────────────────",
            "mkdir -p $SCRATCH/raw",
            "mkdir -p $SCRATCH/manifests/per_dataset",
            "mkdir -p $SCRATCH/alp_cache",
            "mkdir -p $SCRATCH/checkpoints/current_run",
            "",
            "# ── Stage datasets ────────────────────────────────────────────",
        ]

        for anatomy, ds_id in datasets:
            src = f"$STORE/raw/{anatomy}/{ds_id}/"
            dst = f"$SCRATCH/raw/{anatomy}/{ds_id}/"
            lines.append(f'echo "Staging {ds_id} ..."')
            lines.append(f"mkdir -p {dst}")

            # Set Lustre striping BEFORE copying for video datasets
            if ds_id in video_datasets:
                lines.append(
                    f"lfs setstripe -c {lustre_stripe_count} {dst} 2>/dev/null || true"
                )

            lines.append(
                f"rsync -ah --progress {src} {dst} && echo '  ✓ {ds_id}' || "
                f"echo '  ✗ {ds_id} (not found in Store — skipping)'"
            )
            lines.append("")

        lines += [
            "# ── Stage manifests ──────────────────────────────────────────",
            "rsync -avh $STORE/manifests/ $SCRATCH/manifests/ 2>/dev/null || true",
            "",
            "echo 'All staging complete.'",
            "",
            "# ── Post-stage checks ────────────────────────────────────────",
            "echo 'Scratch quota:'",
            f"lfs quota -u $USER /capstor/scratch/ 2>/dev/null || df -h $SCRATCH",
            "",
            "echo 'Staged dataset sizes:'",
        ]

        for anatomy, ds_id in datasets:
            dst = f"$SCRATCH/raw/{anatomy}/{ds_id}"
            lines.append(f'[ -d {dst} ] && du -sh {dst} || echo "{ds_id}: not present"')

        return "\n".join(lines)

    def generate_archive_script(self, phase: int = 1) -> str:
        """Generate a bash script to archive Scratch back to Store after training."""
        return textwrap.dedent(f"""
            #!/bin/bash
            # archive_to_store.sh — archive Phase {phase} outputs back to permanent Store
            STORE="{self.store_root}"
            SCRATCH="{self.scratch_root}"

            set -euo pipefail

            echo "Archiving Phase {phase} checkpoint ..."
            mkdir -p $STORE/checkpoints/phase{phase}
            rsync -avh --progress \\
                $SCRATCH/checkpoints/current_run/ \\
                $STORE/checkpoints/phase{phase}/

            echo "Archiving ALP cache ..."
            rsync -avh --progress \\
                $SCRATCH/alp_cache/ \\
                $STORE/alp_cache/

            echo "Archiving manifests ..."
            rsync -avh --progress \\
                $SCRATCH/manifests/ \\
                $STORE/manifests/

            echo "Archive complete."
        """).strip()

    def generate_touch_script(self) -> str:
        """
        Generate a bash command to reset Scratch access times.
        Prevents files from being purged by the 30-day inactivity policy.
        """
        return (
            f"find {self.scratch_root}/raw/ -type f -exec touch {{}} +"
            f"  # Reset access times to prevent Scratch purge"
        )

    # ── Quota / diagnostics ───────────────────────────────────────────────────

    def check_scratch_quota(self) -> Optional[str]:
        """Run lfs quota on Scratch and return the output string, or None."""
        try:
            user = os.environ.get("USER", "unknown")
            result = subprocess.run(
                ["lfs", "quota", "-u", user, "/capstor/scratch/"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout + result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def summarize(self) -> str:
        """Human-readable summary of configured paths."""
        lines = [
            "CSCSConfig",
            f"  Store  : {self.store_root}  (exists={self.store_root.exists()})",
            f"  Scratch: {self.scratch_root}  (exists={self.scratch_root.exists()})",
            f"  ALP cache: {self.alp_cache_dir()}",
            f"  prefer_scratch: {self.prefer_scratch}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CSCSConfig(store={self.store_root}, "
            f"scratch={self.scratch_root})"
        )


# ── Convenience for data_config.yaml ─────────────────────────────────────────

def resolve_manifest_paths(
    cfg: dict,
    cscs: Optional[CSCSConfig] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve train and val manifest paths from a data_config dict,
    with optional CSCS path resolution.

    Returns (train_path, val_path).
    """
    manifest_cfg = cfg.get("manifest", {})
    train_raw = manifest_cfg.get("path", "")
    val_raw   = manifest_cfg.get("val_path", None)

    if cscs is None:
        return Path(train_raw), Path(val_raw) if val_raw else None

    # Let CSCS prefer Scratch over Store
    def _resolve(raw: str) -> Path:
        p = Path(raw)
        name = p.name
        return cscs.manifest_path(name)

    return _resolve(train_raw), _resolve(val_raw) if val_raw else None


# ── Default anatomy → dataset mapping (for staging script generation) ─────────

DEFAULT_ANATOMY_MAP: Dict[str, str] = {
    # dataset_id: anatomy_folder
    "CAMUS":            "cardiac",
    "EchoNet-Dynamic":  "cardiac",
    "EchoNet-LVH":      "cardiac",
    "MIMIC-IV-ECHO":    "cardiac",
    "COVIDx-US":        "lung",
    "LUS-multicenter-2025": "lung",
    "POCUS-LUS":        "lung",
    "BUS-BRA":          "breast",
    "BUSI":             "breast",
    "BrEaST":           "breast",
    "BUS-UC":           "breast",
    "TN3K":             "thyroid",
    "TN5000":           "thyroid",
    "DDTI":             "thyroid",
    "TNSCUI":           "thyroid",
    "FETAL_PLANES_DB":  "fetal",
    "HC18":             "fetal",
    "ACOUSLIC-AI":      "fetal",
    "KidneyUS":         "kidney",
    "GBCU":             "liver",
    "MMOTU-2D":         "ovarian",
    "PCOSGen":          "ovarian",
    "ASUS":             "musculoskeletal",
    "FALLMUD":          "musculoskeletal",
    "STMUS-NDA":        "musculoskeletal",
}
