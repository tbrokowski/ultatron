"""
finetune/__init__.py
=========================
Dataset-specific finetune experiments.

Each module is a self-contained, runnable experiment:

    python -m finetune.experiments.camus   --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.busi    --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.echonet --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.tn3k    --checkpoint ...  --data-root ...  --config ...
    python -m finetune.experiments.busbra  --checkpoint ...  --data-root ...  --config ...

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

Experiment classes are imported lazily so ``from finetune.video_regression
import ...`` (used by eval benchmarks) does not create circular imports
with eval.benchmarks.echonet when DataLoader workers spawn.
"""
from __future__ import annotations

_LAZY_EXPORTS = {
    "FinetuneExperiment": (".base", "FinetuneExperiment"),
    "FinetuneConfig":     (".base", "FinetuneConfig"),
    "CAMUSFinetune":   (".experiments.camus",   "CAMUSFinetune"),
    "BUSIFinetune":    (".experiments.busi",    "BUSIFinetune"),
    "EchoNetFinetune": (".experiments.echonet", "EchoNetFinetune"),
    "TN3KFinetune":    (".experiments.tn3k",    "TN3KFinetune"),
    "BUSBRAFinetune":  (".experiments.busbra",  "BUSBRAFinetune"),
}

__all__ = [
    "FinetuneExperiment", "FinetuneConfig",
    *list(_LAZY_EXPORTS),
    "EXPERIMENT_REGISTRY",
]


def __getattr__(name: str):
    if name == "EXPERIMENT_REGISTRY":
        return {
            "camus":   __getattr__("CAMUSFinetune"),
            "busi":    __getattr__("BUSIFinetune"),
            "echonet": __getattr__("EchoNetFinetune"),
            "tn3k":    __getattr__("TN3KFinetune"),
            "busbra":  __getattr__("BUSBRAFinetune"),
        }
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod_name, attr = _LAZY_EXPORTS[name]
    mod = importlib.import_module(mod_name, __name__)
    return getattr(mod, attr)
