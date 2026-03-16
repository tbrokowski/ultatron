"""
finetune/__init__.py
=========================
Dataset-specific finetune experiments.

Each module is a self-contained, runnable experiment:

    python -m finetune.experiments.camus   --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.busi    --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.echonet --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.tn3k    --checkpoint ...  --data-root ...  --config ...

Or import and use programmatically:

    from finetune import CAMUSFinetune, FinetuneConfig

    cfg        = FinetuneConfig.from_yaml("configs/finetune/camus.yaml")
    experiment = CAMUSFinetune(data_root=..., output_dir=..., cfg=cfg)
    experiment.setup(img_branch, device="cuda")
    experiment.run()
    results = experiment.evaluate("test")

Design: heads are NOT dataset-specific subclasses.  Each experiment
instantiates the generic head from models/heads/ with the correct
parameters for that dataset (n_classes, head_type, etc.).
Dataset-specific knowledge lives only in the dataloader and loss function.
"""
from .base                 import FinetuneExperiment, FinetuneConfig
from .experiments.camus   import CAMUSFinetune
from .experiments.busi    import BUSIFinetune
from .experiments.echonet import EchoNetFinetune
from .experiments.tn3k    import TN3KFinetune

EXPERIMENT_REGISTRY = {
    "camus":   CAMUSFinetune,
    "busi":    BUSIFinetune,
    "echonet": EchoNetFinetune,
    "tn3k":    TN3KFinetune,
}

__all__ = [
    "FinetuneExperiment", "FinetuneConfig",
    "CAMUSFinetune", "BUSIFinetune", "EchoNetFinetune", "TN3KFinetune",
    "EXPERIMENT_REGISTRY",
]
