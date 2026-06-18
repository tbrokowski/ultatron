"""
data/adapters/cardiac/mimic_echoqa.py  ·  MIMIC-EchoQA VQA adapter
===================================================================

MIMIC-EchoQA: 622 clinician-validated echocardiogram VQA pairs derived from
MIMIC-IV-ECHO.  Each entry pairs one MP4 clip with a closed-ended multiple-choice
question grounded in the echo report and video view.

Actual layout after download (PhysioNet tarball or wget mirror):

  {root}/
    README.md
    MIMICEchoQA/
      MIMICEchoQA.json          ← 622 Q/A records
      0.1/
        files/
          p{prefix}/p{subject_id}/s{study_id}/{study_id}_{series}.mp4

JSON video paths use the MIMIC-IV-ECHO prefix, e.g.:
  mimic-iv-echo/0.1/files/p10/p10119872/s98097718/98097718_0047.mp4

The adapter remaps those to the local MIMICEchoQA/0.1/files/... tree and
emits only entries whose MP4 is present on disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_OPTION_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

_JSON_CANDIDATES = (
    "MIMICEchoQA/MIMICEchoQA.json",
    "physionet.org/files/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json",
    "files/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json",
)


class MIMICEchoQAAdapter(BaseAdapter):
    """
    MIMIC-EchoQA adapter — JSON-driven VQA manifest entries.

    Yields one video entry per Q/A pair in MIMICEchoQA.json.  Missing MP4s are
    skipped so partial downloads produce trainable partial manifests.
    """

    DATASET_ID     = "MIMIC-EchoQA"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://physionet.org/content/mimic-iv-ext-echoqa/"

    def _locate_json(self) -> Tuple[Path, Path]:
        """
        Return (json_path, qa_dir) where qa_dir is the MIMICEchoQA folder
        containing 0.1/files/.
        """
        for rel in _JSON_CANDIDATES:
            candidate = self.root / rel
            if candidate.exists():
                return candidate, candidate.parent

        raise FileNotFoundError(
            f"MIMIC-EchoQA: MIMICEchoQA.json not found under {self.root}.\n"
            "Expected one of:\n"
            + "\n".join(f"  {self.root / rel}" for rel in _JSON_CANDIDATES)
        )

    @staticmethod
    def _resolve_video_path(video_rel: str, qa_dir: Path) -> Path:
        """
        Map a JSON video path to the on-disk MP4 under qa_dir/0.1/files/.

        Handles:
          mimic-iv-echo/0.1/files/p10/...
          MIMICEchoQA/0.1/files/p10/...
          files/p10/...
        """
        rel = video_rel.lstrip("/")
        if rel.startswith("mimic-iv-echo/0.1/"):
            suffix = rel[len("mimic-iv-echo/0.1/"):]
        elif rel.startswith("MIMICEchoQA/0.1/"):
            suffix = rel[len("MIMICEchoQA/0.1/"):]
        elif rel.startswith("0.1/"):
            suffix = rel[len("0.1/"):]
        elif rel.startswith("files/"):
            suffix = rel
        else:
            suffix = rel

        return qa_dir / "0.1" / suffix

    def iter_entries(self) -> Iterator[USManifestEntry]:
        json_path, qa_dir = self._locate_json()
        records = json.loads(json_path.read_text())
        if not isinstance(records, list):
            raise ValueError(
                f"MIMIC-EchoQA: expected a JSON list in {json_path}, "
                f"got {type(records).__name__}"
            )

        n = len(records)
        log.info("MIMIC-EchoQA: %d records in %s", n, json_path.name)

        emitted = 0
        missing = 0
        for i, row in enumerate(records):
            videos = row.get("videos") or []
            if not videos:
                missing += 1
                continue

            video_path = self._resolve_video_path(videos[0], qa_dir)
            if not video_path.exists():
                missing += 1
                continue

            messages_id = row.get("messages_id", "")
            study_id    = row.get("study", "")
            image_id    = row.get("image", video_path.stem)
            answer      = row.get("answer", "")
            correct_opt = str(row.get("correct_option", "")).strip().upper()
            structure   = row.get("structure", "")
            view        = row.get("view", "")

            split = self.split_override or row.get("split") or self._infer_split(
                messages_id or str(video_path), i, n
            )

            options = {
                key[-1]: row.get(key, "")
                for key in ("option_A", "option_B", "option_C", "option_D")
            }
            correct_idx: Optional[int] = _OPTION_INDEX.get(correct_opt)

            instance = self._make_instance(
                instance_id=f"{messages_id}_answer" if messages_id else f"{image_id}_answer",
                label_raw=answer,
                label_ontology=structure.lower().replace(" ", "_") if structure else "cardiac_finding",
                classification_label=correct_idx,
                is_promptable=True,
            )

            emitted += 1
            yield self._make_entry(
                str(video_path),
                split,
                modality           = "video",
                instances          = [instance],
                study_id           = study_id,
                series_id          = messages_id or image_id,
                view_type          = view or None,
                is_cine            = True,
                has_temporal_order = True,
                fps                = 30.0,
                task_type          = "classification",
                ssl_stream         = "video",
                is_promptable      = True,
                label_raw          = [answer] if answer else None,
                source_meta        = {
                    "messages_id":    messages_id,
                    "question":       row.get("question", ""),
                    "answer":         answer,
                    "correct_option": correct_opt,
                    "options":        options,
                    "option_A":       row.get("option_A", ""),
                    "option_B":       row.get("option_B", ""),
                    "option_C":       row.get("option_C", ""),
                    "option_D":       row.get("option_D", ""),
                    "structure":      structure,
                    "report":         row.get("report", ""),
                    "image_id":       image_id,
                    "study_id":       study_id,
                    "view":           view,
                    "video_rel":      videos[0],
                },
            )

        if missing:
            log.warning(
                "MIMIC-EchoQA: skipped %d records with missing video; emitted %d.",
                missing,
                emitted,
            )
        log.info("MIMIC-EchoQA: emitted %d manifest entries", emitted)
