"""
models/video_backbones/__init__.py
Import all video backbone modules so @register_* decorators fire.
"""
from . import vjepa2   # noqa: F401  — registers vjepa2_l/h/g
