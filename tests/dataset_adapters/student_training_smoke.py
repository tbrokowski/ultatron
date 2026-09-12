"""
Backward-compatible shim.

Production entry point: ``python -m train.student_pretrain``
"""
from train.student_pretrain import *  # noqa: F401,F403
from train import student_pretrain as _impl
import sys as _sys

_sys.modules[__name__] = _impl

if __name__ == "__main__":
    _impl.main()
