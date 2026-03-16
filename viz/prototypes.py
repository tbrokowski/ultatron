"""
viz/prototypes.py  ·  Prototype assignment visualisation
=============================================================

Visualises the K learnable prototypes from PrototypeHead.

Three plots:
  1. Prototype map        — colour each spatial patch by its argmax prototype.
                            Reveals how the backbone segments the image semantically.
  2. Prototype confidence — entropy of prototype assignment distribution per patch.
                            Low entropy = confident prototype assignment.
  3. Prototype gallery    — UMAP of all K prototype vectors, coloured by anatomy
                            family when anatomy labels are available.
  4. Assignment histogram — how many patches are assigned to each prototype
                            across a set of images (reveals dead prototypes).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from viz.core import (
    to_numpy_image, tokens_to_spatial, scalar_map_to_heatmap,
    extract_patch_features, save_figure, _get_plt,
)


def compute_prototype_assignments(
    patch_tokens:  torch.Tensor,    # (N, D) or (B, N, D)
    prototypes:    torch.Tensor,    # (K, D)  learnable prototype vectors
    temperature:   float = 0.1,
) -> torch.Tensor:
    """
    Compute soft prototype assignment for each patch token.

    Returns (N, K) or (B, N, K) float32 assignment probabilities.
    """
    squeeze = patch_tokens.ndim == 2
    if squeeze:
        patch_tokens = patch_tokens.unsqueeze(0)

    feat   = F.normalize(patch_tokens.float(), dim=-1)
    proto  = F.normalize(prototypes.float(), dim=-1)
    logits = feat @ proto.T          # (B, N, K)
    probs  = F.softmax(logits / temperature, dim=-1)

    return probs[0] if squeeze else probs


def plot_prototype_map(
    img_branch,
    proto_head,
    image_rgb:   torch.Tensor,      # (3, H, W)
    device:      str   = "cuda",
    temperature: float = 0.1,
    n_colours:   int   = 20,        # how many distinct prototype colours
    alpha:       float = 0.65,
    save_path:   Optional[str] = None,
):
    """
    Colour each patch by its argmax prototype assignment.

    Different prototypes → different colours, revealing semantic partitioning.
    Uses a categorical colourmap so visually similar patches are same colour.

    Returns matplotlib Figure.
    """
    plt = _get_plt()

    feats = extract_patch_features(img_branch, image_rgb, device)
    patch_tokens = feats["patch_tokens"].to(device)
    ph, pw       = feats["ph"], feats["pw"]
    img_np       = to_numpy_image(image_rgb)
    H, W         = img_np.shape[:2]

    prototypes  = proto_head.prototypes.to(device)
    probs       = compute_prototype_assignments(
        patch_tokens, prototypes, temperature
    ).cpu().numpy()                  # (N, K)

    argmax_map  = probs.argmax(-1).reshape(ph, pw)      # (ph, pw)
    entropy_map = -(probs * np.log(probs + 1e-8)).sum(-1).reshape(ph, pw)

    # Categorical colour map: cycle through n_colours
    import matplotlib.cm as cm
    colour_cycle = cm.get_cmap("tab20", n_colours)

    rgb_map = colour_cycle(argmax_map % n_colours)[:, :, :3]   # (ph, pw, 3) [0,1]
    rgb_map = (rgb_map * 255).astype(np.uint8)

    # Upsample prototype map to image resolution
    proto_t = torch.from_numpy(rgb_map).float().permute(2, 0, 1).unsqueeze(0)
    proto_t = F.interpolate(proto_t, size=(H, W), mode="nearest").squeeze(0)
    proto_up = proto_t.permute(1, 2, 0).numpy().astype(np.uint8)

    blended = ((1 - alpha) * img_np.astype(np.float32)
               + alpha     * proto_up.astype(np.float32)
               ).clip(0, 255).astype(np.uint8)

    # Entropy map
    entropy_heat = scalar_map_to_heatmap(entropy_map, cmap="magma")
    entropy_t    = torch.from_numpy(entropy_heat).float().permute(2, 0, 1).unsqueeze(0)
    entropy_up   = F.interpolate(entropy_t, size=(H, W), mode="bilinear", align_corners=False)
    entropy_up   = entropy_up.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    entropy_blend= ((1 - 0.5) * img_np.astype(np.float32)
                    + 0.5 * entropy_up.astype(np.float32)).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title(f"Prototype assignment map\n({prototypes.shape[0]} prototypes)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(entropy_blend)
    axes[2].set_title("Assignment entropy\n(low = confident)", fontsize=10)
    axes[2].axis("off")

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_prototype_umap(
    proto_head,
    labels:   Optional[np.ndarray] = None,      # (K,) int cluster labels
    names:    Optional[list[str]]  = None,       # list of K prototype names
    save_path: Optional[str]       = None,
):
    """
    UMAP of all K prototype vectors coloured by label (if available).

    Reveals clustering structure in the prototype space — prototypes that
    encode similar anatomy should cluster together.
    """
    plt = _get_plt()

    try:
        import umap as umap_lib
    except ImportError:
        try:
            from sklearn.manifold import TSNE as _DR
            use_tsne = True
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Install umap-learn or scikit-learn for this plot",
                    ha="center", va="center")
            return fig
    else:
        use_tsne = False

    protos = F.normalize(proto_head.prototypes.float(), dim=-1).detach().cpu().numpy()
    K, D   = protos.shape

    if not use_tsne:
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15)
        coords  = reducer.fit_transform(protos)
        method  = "UMAP"
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42).fit_transform(protos)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_kwargs = dict(s=40, alpha=0.8, linewidths=0.3, edgecolors="white")

    if labels is not None:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                        cmap="tab20", **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label="Prototype cluster")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(K),
                   cmap="viridis", **scatter_kwargs)

    ax.set_title(f"Prototype vectors — {method} ({K} prototypes, D={D})", fontsize=11)
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_prototype_assignment_histogram(
    img_branch,
    proto_head,
    dataloader,
    device:      str   = "cuda",
    max_batches: int   = 50,
    temperature: float = 0.1,
    save_path:   Optional[str] = None,
):
    """
    Show how often each prototype is the argmax assignment across a dataset.
    Reveals 'dead prototypes' (never assigned) and 'dominant prototypes'.
    """
    plt = _get_plt()

    K      = proto_head.prototypes.shape[0]
    counts = np.zeros(K, dtype=np.int64)

    img_branch.teacher.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            imgs    = batch["global_crops"][:, 0].to(device)
            feats   = img_branch.forward_teacher(imgs)
            tokens  = feats["patch_tokens"]           # (B, N, D)
            probs   = compute_prototype_assignments(
                tokens.reshape(-1, tokens.shape[-1]),
                proto_head.prototypes.to(device),
                temperature,
            ).cpu().numpy()                           # (B*N, K)
            argmax  = probs.argmax(-1)
            for k in argmax:
                counts[k] += 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(K), counts, color="steelblue", alpha=0.8, width=0.9)
    ax.axhline(counts.mean(), color="red", linestyle="--",
               label=f"Mean ({counts.mean():.0f})")
    n_dead = (counts == 0).sum()
    ax.set_title(f"Prototype assignment frequency  "
                 f"({n_dead}/{K} dead prototypes)", fontsize=11)
    ax.set_xlabel("Prototype index")
    ax.set_ylabel("# patch assignments")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
