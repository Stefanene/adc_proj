from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from datasplit import make_split
from preprocess import build_tensors, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, SEED, AUG_MULTIPLIERS
from model import vit_small


# ----------------------------------------------------------------------
# Training config
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = BATCH_SIZE
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    vv_attn_last_n_blocks: int = 1
    num_workers: int = NUM_WORKERS

    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint_dir: Path = Path("./checkpoints")
    log_path: Path = Path("./checkpoints/train_log.json")


# ----------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ----------------------------------------------------------------------
def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # Cosine decay from 1.0 down to 0 over the remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------
# Single epoch — train and eval
# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: str,
    grad_clip: float,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / total_seen:.4f}",
            acc=f"{total_correct / total_seen:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int = NUM_CLASSES,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    return total_loss / total_seen, total_correct / total_seen, confusion


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------
# compute per-class precision, recall, F1 from confusion matrix
def per_class_report(confusion: np.ndarray) -> dict[str, list[float]]:
    n = confusion.shape[0]
    precision, recall, f1 = [], [], []
    for c in range(n):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-12, p + r)
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(f))
    return {"precision": precision, "recall": recall, "f1": f1}


# pretty-print confusion matrix
def print_confusion(confusion: np.ndarray) -> None:
    n = confusion.shape[0]
    header = "      " + "  ".join(f"p{c + 1:>3d}" for c in range(n))
    print(header)
    for i in range(n):
        row = "  ".join(f"{confusion[i, j]:>4d}" for j in range(n))
        print(f"t{i + 1}  {row}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    cfg = TrainConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- Data ----
    train_images, train_labels, val_images, val_labels, _ = make_split(AUG_MULTIPLIERS)
    

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
    print(f"\nModel: ViT-Small  ({n_params / 1e6:.2f}M trainable params)")
    print(f"Device: {cfg.device}")
    print(f"V-V attention in last {cfg.vv_attn_last_n_blocks} block(s)")

    # ---- Loss, optimizer, scheduler ----
    # Class weights: inverse frequency on the original (pre-augmentation) distribution
    # would be ideal, but since augmentation already balances we keep this simple.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ---- Training loop ----
    log: list[dict] = []
    best_val_acc = 0.0
    best_path = cfg.checkpoint_dir / "best.pt"

    print(f"\nStarting training: {cfg.epochs} epochs, {len(train_loader)} steps/epoch\n")
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            cfg.device, cfg.grad_clip, epoch,
        )
        val_loss, val_acc, confusion = evaluate(
            model, val_loader, criterion, cfg.device, NUM_CLASSES,
        )

        elapsed = time.time() - start
        print(
            f"epoch {epoch:>2d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed / 60:.1f}m)"
        )

        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class": per_class_report(confusion),
        })

        # persist log after each epoch so partial runs aren't lost
        with cfg.log_path.open("w") as f:
            json.dump({"config": {k: str(v) for k, v in asdict(cfg).items()}, "log": log}, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "confusion": confusion.tolist(),
            }, best_path)
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {best_path}")

    # ---- Final evaluation report on the best checkpoint ----
    print(f"\nTraining complete in {(time.time() - start) / 60:.1f} minutes")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, final_confusion = evaluate(model, val_loader, criterion, cfg.device, NUM_CLASSES)
    report = per_class_report(final_confusion)

    print(f"\nFinal val accuracy (best ckpt): {final_acc:.4f}")
    print("\nConfusion matrix (rows = true class, cols = predicted):")
    print_confusion(final_confusion)
    print("\nPer-class metrics (classes shown in CSV 1..6 convention):")
    for c in range(NUM_CLASSES):
        print(
            f"  class {c + 1}: "
            f"precision={report['precision'][c]:.4f}  "
            f"recall={report['recall'][c]:.4f}  "
            f"f1={report['f1'][c]:.4f}"
        )


if __name__ == "__main__":
    main()