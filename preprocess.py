
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


from visualize_tensor import visualize_tensors, visualize_one_per_class, show_tensor_stats


DATA_ROOT = Path("./data")
METADATA_CSV = DATA_ROOT / "carinthia.csv"
IMAGES_ROOT = DATA_ROOT / "images"
# CSV columns
PATH_COL = "image_path"
FILENAME_COL = "file_name"
LABEL_COL = "label"
# Labels in the CSV are 1..6; we map them to 0..5 internally for PyTorch.
NUM_CLASSES = 6
IMAGE_SIZE = 224   # ViT-base default
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
VAL_SPLIT = 0.2
# Per-label augmentation multiplier: label (0-indexed) -> how many times to
# multiply that class's sample count.  A value of 1 means no augmentation.
AUG_MULTIPLIERS: dict[int, int] = {
    0: 5,
    1: 30,
    2: 1,
    3: 1,
    4: 50,
    5: 1,
}
# SEM images are grayscale, replicated to 3 channels so that pretrained
# ImageNet weights remain meaningful.
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


def augment_per_label(images: torch.Tensor,
                      labels: torch.Tensor,
                      multipliers: dict[int, int] = AUG_MULTIPLIERS,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    # types of augmentations
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
    ])

    aug_images = []
    aug_labels = []

    for cls in range(NUM_CLASSES):
        mult = multipliers.get(cls, 1)
        if mult <= 1:
            continue

        cls_mask = labels == cls
        cls_images = images[cls_mask]
        n_existing = cls_images.shape[0]
        n_needed = n_existing * (mult - 1)

        sample_idx = torch.randint(0, n_existing, (n_needed,))
        for idx in sample_idx:
            aug_images.append(aug(cls_images[idx]))
            aug_labels.append(cls)

    if aug_images:
        aug_images = torch.stack(aug_images)
        aug_labels = torch.tensor(aug_labels, dtype=torch.long)
        images = torch.cat([images, aug_images], dim=0)
        labels = torch.cat([labels, aug_labels], dim=0)

    print(f"After augmentation: {images.shape[0]} samples")
    print(f"Augmented class counts: {torch.bincount(labels, minlength=NUM_CLASSES).tolist()}")

    return images, labels


def build_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(METADATA_CSV, sep=';')

    # resize is required for stacking and for ViT input size
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img_list = []
    label_list = []

    for path, label in tqdm(zip(df[PATH_COL], df[LABEL_COL]), total=len(df), desc="Loading data"):
        img = Image.open(path).convert("L").convert("RGB")

        img_list.append(transform(img))
        label_list.append(int(label)-1)     # rescaled 0-5

    img_tensor = torch.stack(img_list, dim=0)
    label_tensor = torch.tensor(label_list, dtype=torch.long)

    print(f"\tImages: {img_tensor.shape}, dtype={img_tensor.dtype}")
    print(f"\tLabels: {label_tensor.shape}, dtype={label_tensor.dtype}")
    print(f"\tLabel range: [{label_tensor.min().item()}, {label_tensor.max().item()}]")
    print(f"\tClass counts: {torch.bincount(label_tensor, minlength=NUM_CLASSES).tolist()}")

    # Augment minority classes before splitting
    img_tensor, label_tensor = augment_per_label(img_tensor, label_tensor)

    # Train/val split
    n = len(label_tensor)
    indices = torch.randperm(n)
    val_size = int(n * VAL_SPLIT)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_images, train_labels = img_tensor[train_idx], label_tensor[train_idx]
    val_images, val_labels = img_tensor[val_idx], label_tensor[val_idx]

    print(f"\nTrain: {train_images.shape[0]} samples")
    print(f"Val:   {val_images.shape[0]} samples")
    print(f"Train class counts: {torch.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")
    print(f"Val   class counts: {torch.bincount(val_labels, minlength=NUM_CLASSES).tolist()}")

    return train_images, train_labels, val_images, val_labels


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_images, train_labels, val_images, val_labels = build_tensors()

    # # Quick numerical sanity check
    # show_tensor_stats(train_images, train_labels)

    # # First 8 images in a 2x4 grid
    visualize_tensors(train_images, train_labels, n=8, ncols=4, title="First 8 images")

    # # One image per class — checks that your label mapping is correct
    # visualize_one_per_class(train_images, train_labels, num_classes=NUM_CLASSES)

    # # Specific indices, e.g. inspect a suspicious example
    # visualize_tensors(train_images, train_labels, indices=[0, 42, 100, 500], ncols=4)