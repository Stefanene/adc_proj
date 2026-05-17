import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_tensors(
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    indices: list[int] | None = None,
    n: int = 8,
    ncols: int = 4,
    title: str | None = None,
) -> None:
    """
    Visualize SEM image tensors in a grid.

    Args:
        images: tensor of shape (N, 3, H, W), values in [0, 1].
        labels: optional tensor of shape (N,) with 0-indexed class labels.
                Displayed as 1-indexed to match the CSV convention.
        indices: specific indices to visualize. If None, picks the first `n`.
        n: number of images to show when `indices` is None.
        ncols: number of columns in the grid.
        title: optional figure-level title.
    """
    if indices is None:
        indices = list(range(min(n, images.size(0))))

    n = len(indices)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(-1)  # flatten for uniform indexing

    for ax_i, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img = np.clip(img, 0, 1)

        axes[ax_i].imshow(img)
        if labels is not None:
            axes[ax_i].set_title(f"idx {idx} — class {int(labels[idx]) + 1}", fontsize=9)
        else:
            axes[ax_i].set_title(f"idx {idx}", fontsize=9)
        axes[ax_i].axis("off")

    # hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.show()


def visualize_one_per_class(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 6,
) -> None:
    
    indices = []
    for c in range(num_classes):
        matches = (labels == c).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            indices.append(int(matches[0]))

    visualize_tensors(
        images, labels,
        indices=indices,
        ncols=num_classes,
        title="One example per class",
    )


def show_tensor_stats(images: torch.Tensor, labels: torch.Tensor) -> None:
    """Print a quick numerical sanity check of the tensors."""
    print(f"Images: shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"  value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  per-channel mean: {images.mean(dim=(0, 2, 3)).tolist()}")
    print(f"  per-channel std:  {images.std(dim=(0, 2, 3)).tolist()}")
    print(f"Labels: shape={tuple(labels.shape)}, dtype={labels.dtype}")
    print(f"  class counts: {torch.bincount(labels, minlength=6).tolist()}")