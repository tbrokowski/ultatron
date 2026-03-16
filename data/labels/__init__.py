"""oura.data.labels — label schema shared with models/."""
from .label_spec import TaskType, LossType, LabelSpec
from .label_interface import LabelTarget, HeadRegistry

__all__ = ["TaskType", "LossType", "LabelSpec", "LabelTarget", "HeadRegistry"]
