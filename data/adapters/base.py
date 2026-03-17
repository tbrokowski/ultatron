"""
data/adapters/base.py  ·  Base adapter class
==================================================

Every per-dataset adapter inherits from BaseAdapter and implements
iter_entries() to yield USManifestEntry objects.

The adapter is responsible for:
  - Parsing the dataset's directory structure and metadata files
  - Mapping raw label strings to the canonical ontology vocabulary
  - Assigning split (train/val/test) either from dataset metadata or
    by deterministic hash-based splitting
  - Constructing Instance objects for each labelled structure
  - Setting task_type, ssl_stream, and is_promptable flags correctly

Adding a new dataset
--------------------
1. Create oura/data/adapters/{dataset_name}.py
2. Subclass BaseAdapter, set DATASET_ID, ANATOMY_FAMILY, SONODQS
3. Implement iter_entries()
4. Register in oura/data/adapters/registry.py

SonoDQS quality tiers
---------------------
  platinum  : prospective, multi-centre, DICOM, expert labels, public
  gold      : public dataset, well-curated, standard labels
  silver    : public dataset, some noise or partial labels
  bronze    : web-scraped or weakly labelled
  unrated   : not assessed
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple

from data.schema.manifest import USManifestEntry, Instance, SONODQS_SCORE


class BaseAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Class attributes to set on each subclass
    -----------------------------------------
    DATASET_ID     : str   canonical dataset identifier (matches manifest)
    ANATOMY_FAMILY : str   primary anatomy (cardiac, breast, thyroid, ...)
    SONODQS        : str   quality tier (platinum/gold/silver/bronze/unrated)
    DEFAULT_SPLIT_RATIO : (train_frac, val_frac, test_frac)
    DOI            : str   dataset DOI or URL for provenance
    """

    DATASET_ID:          str = "unknown"
    ANATOMY_FAMILY:      str = "other"
    SONODQS:             str = "unrated"
    DEFAULT_SPLIT_RATIO: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    DOI:                 str = ""

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        root           : dataset root directory
        split_override : if set, all entries get this split label.
                         Useful for loading a pre-split subset.
        """
        self.root           = Path(root)
        self.split_override = split_override

        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.DATASET_ID}: root not found: {self.root}"
            )

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def iter_entries(self) -> Iterator[USManifestEntry]:
        """
        Yield all USManifestEntry objects for this dataset.

        Implementors should yield entries in a deterministic order
        (sorted by filename or patient ID) for reproducibility.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _make_entry(
        self,
        image_paths: str | list[str],
        split: str,
        modality: str = "image",
        instances: Optional[list] = None,
        **kwargs,
    ) -> USManifestEntry:
        """
        Convenience factory for USManifestEntry.

        Fills in DATASET_ID, ANATOMY_FAMILY, SONODQS, quality_score,
        and source_meta automatically.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Allow adapters to extend source_meta without overriding the defaults.
        extra_source_meta = kwargs.pop("source_meta", {}) or {}
        source_meta = {"root": str(self.root), "doi": self.DOI}
        source_meta.update(extra_source_meta)

        entry = USManifestEntry(
            sample_id=USManifestEntry.make_sample_id(self.DATASET_ID, image_paths[0]),
            dataset_id=self.DATASET_ID,
            anatomy_family=self.ANATOMY_FAMILY,
            sonodqs=self.SONODQS,
            quality_score=SONODQS_SCORE.get(self.SONODQS.lower(), 1),
            split=split,
            modality_type=modality,
            image_paths=image_paths,
            instances=instances or [],
            source_meta=source_meta,
            **kwargs,
        )
        # Assign curriculum tier based on entry properties
        from data.schema.manifest import assign_curriculum_tier
        entry.curriculum_tier = assign_curriculum_tier(entry)
        return entry

    def _make_instance(
        self,
        instance_id: str,
        label_raw: str,
        label_ontology: str,
        mask_path: Optional[str] = None,
        is_promptable: bool = True,
        **kwargs,
    ) -> Instance:
        """Convenience factory for Instance objects."""
        return Instance(
            instance_id=instance_id,
            label_raw=label_raw,
            label_ontology=label_ontology,
            anatomy_family=self.ANATOMY_FAMILY,
            mask_path=mask_path,
            is_promptable=is_promptable,
            **kwargs,
        )

    def _infer_split(self, identifier: str, idx: int, total: int) -> str:
        """
        Deterministic train/val/test assignment.

        Uses modulo of index first; if split_override is set, always returns that.
        """
        if self.split_override:
            return self.split_override

        train_frac, val_frac, _ = self.DEFAULT_SPLIT_RATIO
        frac = idx / max(total - 1, 1)
        if frac < train_frac:
            return "train"
        elif frac < train_frac + val_frac:
            return "val"
        else:
            return "test"

    def build_manifest(self) -> list[USManifestEntry]:
        """Collect all entries into a list. Convenience wrapper."""
        return list(self.iter_entries())

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"dataset={self.DATASET_ID!r}, "
                f"root={self.root})")
