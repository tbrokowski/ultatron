"""
viz/__init__.py
====================
Visualisation module for Ultatron.  Covers attention maps, prototype maps,
feature spaces, segmentation overlays, training diagnostics, regression
analysis, linear probe curves, SAM prompt inspection, and summary dashboards.

All functions produce matplotlib Figure objects and can optionally save to disk.
No side effects on model state — all viz functions operate in torch.no_grad().

Quick start
-----------
    from .viz import attention, prototypes, features, segmentation

    # Attention maps
    fig = attention.plot_attention_maps(img_branch, image_rgb)
    fig.savefig("attention.png", dpi=150)

    # Prototype assignment map
    fig = prototypes.plot_prototype_map(img_branch, proto_head, image_rgb)

    # Feature UMAP coloured by anatomy
    fig = features.plot_umap(features_array, labels, anatomy_families)

    # Segmentation overlay
    fig = segmentation.plot_overlay(image, pred_mask, gt_mask, title="CAMUS 2CH ED")
"""
from . import core
from . import attention
from . import prototypes
from . import features
from . import segmentation
from . import training
from . import regression
from . import probes
from . import sam_prompts
from . import dashboard

__all__ = [
    "core", "attention", "prototypes", "features",
    "segmentation", "training", "regression",
    "probes", "sam_prompts", "dashboard",
]
