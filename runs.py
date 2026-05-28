"""
runs.py
-------
Run the full baseline comparison suite for the SEM defect classification project.

Experiment matrix:
    1. Vanilla ViT-Small         — no V-V, no per-class augmentation
    2. ViT-Small + augmentation  — no V-V, several multiplier configs (ablation)
    3. ViT-Small + V-V + aug     — full proposed method

SEM-CLIP (~90% on a different dataset) is included as an external reference point,
not a run we execute. Our train.py is already achieving ~90% val acc on the full
method, which is the target SEM-CLIP-comparable number.

Results are collected into ./runs/results.json and a summary table is printed.
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reuse the training machinery
from train import (
    TrainConfig,
    build_lr_scheduler,
    train_one_epoch,
    evaluate,
    per_class_report,
    print_confusion,
)
from preprocess import (
    build_tensors,
    augment_per_label,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from model import vit_small
from datasplit import make_split

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
# Multipliers used in your current train.py run (the 90.45% setup)
FULL_AUG_MULTIPLIERS: dict[int, int] = {0: 5, 1: 30, 2: 1, 3: 1, 4: 50, 5: 1}

# No-augmentation baseline — all classes pass through unchanged
NO_AUG_MULTIPLIERS: dict[int, int] = {c: 1 for c in range(NUM_CLASSES)}

# Light augmentation — modest oversampling on minority classes only
LIGHT_AUG_MULTIPLIERS: dict[int, int] = {0: 2, 1: 10, 2: 1, 3: 1, 4: 15, 5: 1}

# Medium augmentation — between light and full
MEDIUM_AUG_MULTIPLIERS: dict[int, int] = {0: 3, 1: 20, 2: 1, 3: 1, 4: 30, 5: 1}


@dataclass
class RunSpec:
    """Specification of a single experimental run."""
    name: str
    description: str
    vv_attn_last_n_blocks: int
    aug_multipliers: dict[int, int]
    epochs: int = 30


# The three-tier baseline structure from the proposal slide
EXPERIMENT_MATRIX: list[RunSpec] = [
    # Tier 1: Vanilla baseline — no V-V, no augmentation
    RunSpec(
        name="vanilla",
        description="Vanilla ViT-Small (no V-V, no aug)",
        vv_attn_last_n_blocks=0,
        aug_multipliers=NO_AUG_MULTIPLIERS,
    ),

    # Tier 2: Augmentation ablation — no V-V, varying multipliers
    RunSpec(
        name="aug_light",
        description="ViT-Small + light per-class aug (no V-V)",
        vv_attn_last_n_blocks=0,
        aug_multipliers=LIGHT_AUG_MULTIPLIERS,
    ),
    RunSpec(
        name="aug_medium",
        description="ViT-Small + medium per-class aug (no V-V)",
        vv_attn_last_n_blocks=0,
        aug_multipliers=MEDIUM_AUG_MULTIPLIERS,
    ),
    RunSpec(
        name="aug_full",
        description="ViT-Small + full per-class aug (no V-V)",
        vv_attn_last_n_blocks=0,
        aug_multipliers=FULL_AUG_MULTIPLIERS,
    ),

    # Tier 3: Full proposed method — V-V + full augmentation
    RunSpec(
        name="full_method",
        description="ViT-Small + V-V + full per-class aug (ours)",
        vv_attn_last_n_blocks=1,
        aug_multipliers=FULL_AUG_MULTIPLIERS,
    ),
]


# ----------------------------------------------------------------------
# Data preparation per-run (since augmentation differs across runs)
# ----------------------------------------------------------------------
def prepare_data_for_run(
    spec: RunSpec, cached_raw: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load raw images once and re-apply per-run augmentation.

    We cache the unaugmented (image, label) tensors across runs because loading
    4,591 JPEGs from disk repeatedly is wasteful. Each run then applies its
    own augmentation multiplier set to those cached tensors.
    """
    if cached_raw is None:
        # First run: load from disk via build_tensors (without aug yet)
        from preprocess import (
            METADATA_CSV, PATH_COL, LABEL_COL, IMAGE_SIZE, VAL_SPLIT,
        )
        from torchvision import transforms
        from PIL import Image
        import pandas as pd
        from tqdm import tqdm

        df = pd.read_csv(METADATA_CSV, sep=";")
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        img_list, label_list = [], []
        for path, label in tqdm(
            zip(df[PATH_COL], df[LABEL_COL]),
            total=len(df), desc="Loading raw images",
        ):
            img = Image.open(path).convert("L").convert("RGB")
            img_list.append(transform(img))
            label_list.append(int(label) - 1)

        raw_images = torch.stack(img_list, dim=0)
        raw_labels = torch.tensor(label_list, dtype=torch.long)
        cached_raw = (raw_images, raw_labels)

    raw_images, raw_labels = cached_raw

    # Apply per-run augmentation
    aug_images, aug_labels = augment_per_label(
        raw_images.clone(), raw_labels.clone(),
        multipliers=spec.aug_multipliers,
    )

    # Stratified-style split: same seed for all runs so val sets are comparable
    from preprocess import VAL_SPLIT
    n = len(aug_labels)
    g = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=g)
    val_size = int(n * VAL_SPLIT)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        aug_images[train_idx], aug_labels[train_idx],
        aug_images[val_idx], aug_labels[val_idx],
        cached_raw,  # return cache for reuse by next run
    )


# ----------------------------------------------------------------------
# Single-run executor
# ----------------------------------------------------------------------
def run_experiment(
    spec: RunSpec,
    cached_raw: tuple[torch.Tensor, torch.Tensor] | None,
    output_dir: Path,
) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
    """
    Train one configuration end-to-end and return its result dict.

    The result dict includes final val accuracy, per-class metrics, training
    time, and the confusion matrix.
    """
    print(f"\n{'=' * 70}")
    print(f"RUN: {spec.name}")
    print(f"  {spec.description}")
    print(f"  V-V blocks: {spec.vv_attn_last_n_blocks}")
    print(f"  Multipliers: {spec.aug_multipliers}")
    print(f"{'=' * 70}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data
    (train_images, train_labels,
    val_images, val_labels,
    cached_raw) = make_split(spec.aug_multipliers, cached_raw=cached_raw)


    print(f"Train: {train_images.shape[0]} samples, "
          f"class counts: {torch.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")
    print(f"Val:   {val_images.shape[0]} samples, "
          f"class counts: {torch.bincount(val_labels, minlength=NUM_CLASSES).tolist()}")

    cfg = TrainConfig(
        epochs=spec.epochs,
        vv_attn_last_n_blocks=spec.vv_attn_last_n_blocks,
        checkpoint_dir=output_dir / spec.name,
        log_path=output_dir / spec.name / "train_log.json",
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TensorDataset(train_images, train_labels)
    val_ds = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
    )

    # ---- Model ----
    model = vit_small(
        num_classes=NUM_CLASSES,
        vv_attn_last_n_blocks=cfg.vv_attn_last_n_blocks,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params / 1e6:.2f}M params, device {cfg.device}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ---- Train ----
    best_val_acc = 0.0
    best_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    best_epoch = 0
    best_path = cfg.checkpoint_dir / "best.pt"
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            cfg.device, cfg.grad_clip, epoch,
        )
        val_loss, val_acc, confusion = evaluate(
            model, val_loader, criterion, cfg.device, NUM_CLASSES,
        )

        print(
            f"  epoch {epoch:>2d}/{cfg.epochs}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_confusion = confusion
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, best_path)

    elapsed = time.time() - start
    report = per_class_report(best_confusion)

    print(f"\n  done in {elapsed / 60:.1f}m  |  best val_acc={best_val_acc:.4f} "
          f"(epoch {best_epoch})")
    print(f"  best confusion matrix:")
    print_confusion(best_confusion)

    result = {
        "name": spec.name,
        "description": spec.description,
        "vv_attn_last_n_blocks": spec.vv_attn_last_n_blocks,
        "aug_multipliers": spec.aug_multipliers,
        "epochs": spec.epochs,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "train_time_minutes": float(elapsed / 60),
        "n_train": int(train_images.shape[0]),
        "n_val": int(val_images.shape[0]),
        "per_class": report,
        "confusion_matrix": best_confusion.tolist(),
    }
    return result, cached_raw


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
# SEM-CLIP reported number on their in-house 6-class CMOS dataset (for context only)
SEM_CLIP_REFERENCE = {
    "name": "sem_clip_reference",
    "description": "SEM-CLIP reference (different dataset, external)",
    "best_val_acc": 0.90,
    "is_external_reference": True,
}


def print_summary_table(results: list[dict]) -> None:
    """Print a side-by-side comparison of all runs."""
    print(f"\n{'=' * 90}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'=' * 90}")
    print(f"{'run':<16}  {'V-V':<4}  {'aug':<10}  {'val_acc':<10}  "
          f"{'min_class_f1':<14}  {'n_train':<8}")
    print("-" * 90)

    for r in results:
        # Identify minority class F1 — the metric we care most about
        f1s = r["per_class"]["f1"]
        min_f1_idx = int(np.argmin(f1s))
        min_f1 = f1s[min_f1_idx]

        aug_kind = "none" if all(v == 1 for v in r["aug_multipliers"].values()) else \
                   "light" if max(r["aug_multipliers"].values()) <= 15 else \
                   "medium" if max(r["aug_multipliers"].values()) <= 30 else \
                   "full"

        print(f"{r['name']:<16}  "
              f"{r['vv_attn_last_n_blocks']:<4}  "
              f"{aug_kind:<10}  "
              f"{r['best_val_acc']:.4f}    "
              f"{min_f1:.4f} (c{min_f1_idx + 1})  "
              f"{r['n_train']:<8}")

    # External reference
    print("-" * 90)
    print(f"{'sem_clip_ref':<16}  {'--':<4}  {'--':<10}  "
          f"{SEM_CLIP_REFERENCE['best_val_acc']:.4f}    "
          f"{'(external, different dataset)':<25}")
    print(f"{'=' * 90}\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparison suite")
    parser.add_argument("--output-dir", type=Path, default=Path("./runs"),
                        help="Directory for run outputs and results.json")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated run names to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all runs (useful for smoke tests)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"

    # Resume from prior results if present
    results: list[dict] = []
    if results_path.exists():
        with results_path.open() as f:
            results = json.load(f)["results"]
        completed_names = {r["name"] for r in results}
        print(f"Resuming: {len(completed_names)} runs already complete "
              f"({sorted(completed_names)})")
    else:
        completed_names = set()

    # Filter the matrix to what the user requested
    to_run = EXPERIMENT_MATRIX
    if args.only is not None:
        wanted = set(args.only.split(","))
        to_run = [s for s in to_run if s.name in wanted]
    if args.epochs is not None:
        to_run = [RunSpec(**{**asdict(s), "epochs": args.epochs}) for s in to_run]

    cached_raw = None  # (raw_images, raw_labels) reused across runs

    suite_start = time.time()
    for spec in to_run:
        if spec.name in completed_names:
            print(f"\nSkipping {spec.name} (already complete in results.json)")
            continue

        result, cached_raw = run_experiment(spec, cached_raw, args.output_dir)
        results.append(result)

        # Persist after every run so a crash doesn't lose progress
        with results_path.open("w") as f:
            json.dump({"results": results, "reference": SEM_CLIP_REFERENCE}, f, indent=2)
        print(f"  results saved to {results_path}")

    print(f"\nAll runs complete in {(time.time() - suite_start) / 60:.1f}m total")
    print_summary_table(results)


if __name__ == "__main__":
    main()