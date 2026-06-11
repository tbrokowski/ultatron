"""
data/adapters/abdomen/abdomen_us_liver.py  ·  AbdomenUS-liver adapter
======================================================================

Mis-placed copy of the Abdominal Ultrasound (AUS/RUS) dataset under liver/.
Reuses the AbdomenUS layout parser with a distinct dataset_id.
"""
from __future__ import annotations

from .abdomen_us import AbdomenUSAdapter


class AbdomenUSLiverAdapter(AbdomenUSAdapter):
    DATASET_ID = "AbdomenUS-liver"
