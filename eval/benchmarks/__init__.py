"""
eval/benchmarks/__init__.py
============================
Per-dataset benchmark runners.  Each module runs a complete evaluation
loop for one dataset and returns a standardised results dict.

Available benchmarks
--------------------
  camus         CAMUS LV segmentation (Dice, IoU, Hausdorff-95)
  echonet       EchoNet-Dynamic EF regression (MAE, R², RMSE)
  busi          BUSI breast tumour segmentation (Dice per class)
  tn3k          TN3K thyroid nodule detection/segmentation
  acouslic      ACOUSLIC-AI fetal standard plane detection

All runners share the same interface:
    from eval.benchmarks.camus import CAMUSBenchmark
    results = CAMUSBenchmark(img_branch, seg_head, dm).run()
    # results: {"dice_mean": float, "dice_ed": float, "dice_es": float, ...}

Imports are lazy so submodule imports (e.g. eval.benchmarks.busi) do not
pull in unrelated benchmarks — avoids circular imports when DataLoader
workers spawn on Linux (multiprocessing spawn re-imports from scratch).
"""
from __future__ import annotations

_LAZY_EXPORTS = {
    "CAMUSBenchmark":   (".camus",   "CAMUSBenchmark"),
    "EchoNetBenchmark": (".echonet", "EchoNetBenchmark"),
    "BUSIBenchmark":    (".busi",    "BUSIBenchmark"),
    "TN3KBenchmark":    (".tn3k",    "TN3KBenchmark"),
    "BUSBRABenchmark":  (".busbra",  "BUSBRABenchmark"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod_name, attr = _LAZY_EXPORTS[name]
    mod = importlib.import_module(mod_name, __name__)
    return getattr(mod, attr)
