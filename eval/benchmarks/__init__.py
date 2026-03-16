"""
oura/eval/benchmarks/__init__.py
=================================
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
    from oura.eval.benchmarks.camus import CAMUSBenchmark
    results = CAMUSBenchmark(img_branch, seg_head, dm).run()
    # results: {"dice_mean": float, "dice_ed": float, "dice_es": float, ...}
"""
from .camus    import CAMUSBenchmark
from .echonet  import EchoNetBenchmark
from .busi     import BUSIBenchmark
from .tn3k     import TN3KBenchmark

__all__ = [
    "CAMUSBenchmark",
    "EchoNetBenchmark",
    "BUSIBenchmark",
    "TN3KBenchmark",
]
