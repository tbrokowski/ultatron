"""
models/__init__.py
==================
Import all backbone subpackages so their registrations fire on first import.
Then expose the public API: ModelConfig, build_image_branch, build_video_branch.
"""
# Trigger all @register_* decorators
from . import image_backbones   # noqa: F401
from . import video_backbones   # noqa: F401


from .model_config import (     # noqa: F401
    ModelConfig,
    build_image_branch,
    build_video_branch,
)
from .branches import (         # noqa: F401
    ImageBranch,
    VideoBranch,
    CrossBranchDistillation,
    PrototypeHead,
    ema_update,
)
from .registry import (         # noqa: F401
    list_image_backbones,
    list_video_backbones,
    list_frozen_teachers,
)
