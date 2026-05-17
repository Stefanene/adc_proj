
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
# SEM images are grayscale, replicated to 3 channels so that pretrained
# ImageNet weights remain meaningful.
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


def build_tensors() -> tuple[torch.Tensor, torch.Tensor]:
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

    return img_tensor, label_tensor


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    images, labels = build_tensors()

    # Quick numerical sanity check
    show_tensor_stats(images, labels)

    # First 8 images in a 2x4 grid
    visualize_tensors(images, labels, n=8, ncols=4, title="First 8 images")

    # One image per class — checks that your label mapping is correct
    visualize_one_per_class(images, labels, num_classes=NUM_CLASSES)

    # Specific indices, e.g. inspect a suspicious example
    visualize_tensors(images, labels, indices=[0, 42, 100, 500], ncols=4)