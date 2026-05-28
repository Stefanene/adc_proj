from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from preprocess import (
    METADATA_CSV, PATH_COL, LABEL_COL,
    NUM_CLASSES, IMAGE_SIZE, VAL_SPLIT, SEED,
    augment_per_label,
)


# ----------------------------------------------------------------------
# Raw image loading (no augmentation, no split)
# ----------------------------------------------------------------------
def load_raw_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(METADATA_CSV, sep=";")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img_list, label_list = [], []
    for path, label in tqdm(
        zip(df[PATH_COL], df[LABEL_COL]), total=len(df), desc="Loading raw images"
    ):
        img = Image.open(path).convert("L").convert("RGB")
        img_list.append(transform(img))
        label_list.append(int(label) - 1)

    images = torch.stack(img_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)
    return images, labels


# ----------------------------------------------------------------------
# Stratified split on RAW data (fixed and shared across all runs)
# ----------------------------------------------------------------------
def stratified_raw_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    train_idx_parts, val_idx_parts = [], []
    for c in range(NUM_CLASSES):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0]
        n_c = cls_idx.numel()
        if n_c == 0:
            continue

        perm = cls_idx[torch.randperm(n_c, generator=g)]
        n_val = int(round(n_c * val_split))
        # Guarantee at least 1 train sample if the class has >=2 images
        if n_c >= 2:
            n_val = min(max(n_val, 1), n_c - 1)
        else:
            n_val = 0  # single-sample class -> keep it in train

        val_idx_parts.append(perm[:n_val])
        train_idx_parts.append(perm[n_val:])

    train_idx = torch.cat(train_idx_parts)
    val_idx = torch.cat(val_idx_parts)

    # Shuffle the concatenated train indices (same generator) so classes interleave
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]

    return (
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
    )


# ----------------------------------------------------------------------
# Top-level: produce train (augmented) + val (clean) for a given config
# ----------------------------------------------------------------------
def make_split(
    multipliers: dict[int, int],
    cached_raw: tuple[torch.Tensor, torch.Tensor] | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           tuple[torch.Tensor, torch.Tensor]]:
    
    if cached_raw is None:
        cached_raw = load_raw_tensors()
    raw_images, raw_labels = cached_raw

    # 1) Split RAW data first (val is now fixed, clean, shared)
    train_images, train_labels, val_images, val_labels = stratified_raw_split(
        raw_images, raw_labels,
    )

    if verbose:
        print(f"Raw split -> train {train_images.shape[0]}, val {val_images.shape[0]}")
        print(f"  raw train counts: {torch.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")
        print(f"  val   counts:     {torch.bincount(val_labels, minlength=NUM_CLASSES).tolist()}  (FIXED across all runs)")

    # 2) Augment the TRAIN set only
    if any(m > 1 for m in multipliers.values()):
        train_images, train_labels = augment_per_label(
            train_images.clone(), train_labels.clone(), multipliers=multipliers,
        )

    if verbose:
        print(f"After train-only augmentation -> train {train_images.shape[0]}")
        print(f"  aug train counts: {torch.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")

    return train_images, train_labels, val_images, val_labels, cached_raw


if __name__ == "__main__":
    # Sanity check: the val set must be identical regardless of multipliers
    FULL = {0: 5, 1: 30, 2: 1, 3: 1, 4: 50, 5: 1}
    NONE = {c: 1 for c in range(NUM_CLASSES)}

    _, _, val_a, val_a_lbl, cache = make_split(NONE, verbose=False)
    _, _, val_b, val_b_lbl, _ = make_split(FULL, cached_raw=cache, verbose=False)

    same_size = val_a.shape == val_b.shape
    same_labels = torch.equal(val_a_lbl, val_b_lbl)
    same_pixels = torch.allclose(val_a, val_b)
    print(f"Val sets identical across configs: "
          f"size={same_size}, labels={same_labels}, pixels={same_pixels}")
    print(f"Val class counts: {torch.bincount(val_a_lbl, minlength=NUM_CLASSES).tolist()}")