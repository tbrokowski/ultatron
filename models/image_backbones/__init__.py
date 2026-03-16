"""
models/image_backbones/__init__.py
Import all backbone modules so their @register_* decorators fire
when the package is imported.
"""
from . import dinov3    # noqa: F401  — registers dinov3_s/b/l/hplus + dinov3_7b
from . import rad_dino  # noqa: F401  — registers rad_dino
from . import swin_v2   # noqa: F401  — registers swin_v2_l
