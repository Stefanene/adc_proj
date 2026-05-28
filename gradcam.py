import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

from preprocess import build_tensors, NUM_CLASSES, IMAGE_SIZE, SEED
from model import vit_small


# ----------------------------------------------------------------------
# Grad-CAM core
# ----------------------------------------------------------------------
class ViTGradCAM:

    def __init__(self, model, block_idx: int = -2) -> None:
        self.model = model
        self.model.eval()

        target = model.blocks[block_idx].norm1
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.fwd_handle = target.register_forward_hook(self._save_activation)
        self.bwd_handle = target.register_full_backward_hook(self._save_gradient)

        # spatial grid dimensions: 224 / 16 = 14
        self.grid_size = model.cfg.image_size // model.cfg.patch_size

    def _save_activation(self, module, inputs, output):
        # output: (B, N+1, D) — includes CLS token at index 0
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: (B, N+1, D)
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, image: torch.Tensor, target_class: int | None = None
                 ) -> tuple[np.ndarray, int, float]:
        x = image.unsqueeze(0).requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())
        confidence = float(probs[0, target_class].item())

        # backprop the chosen class score
        self.model.zero_grad()
        logits[0, target_class].backward()

        # drop CLS, reshape to spatial grid
        act = self.activations[0, 1:]                       # (N, D)
        grad = self.gradients[0, 1:]                        # (N, D)

        # channel-wise importance from gradients
        weights = grad.mean(dim=0)                          # (D,)
        cam = (act * weights).sum(dim=-1)                   # (N,)
        cam = cam.reshape(self.grid_size, self.grid_size)   # (14, 14)

        cam = F.relu(cam)
        # upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # normalize to [0, 1] for visualization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def overlay_cam(image: torch.Tensor, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay the heatmap on the image. Returns an (H, W, 3) array in [0, 1]."""
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)

    heatmap = cm.jet(cam)[..., :3]  # (H, W, 3) RGB from colormap

    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def visualize_gradcam_grid(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int] | None = None,
    n: int = 12,
    block_idx: int = -2,
    save_path: Path | None = None,
) -> None:
    if indices is None:
        # pick one image per class for a balanced view
        indices = []
        for c in range(NUM_CLASSES):
            matches = (labels == c).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                indices.append(int(matches[0]))
        # fill up to n with additional samples if needed
        all_idx = list(range(images.size(0)))
        for i in all_idx:
            if len(indices) >= n:
                break
            if i not in indices:
                indices.append(i)

    indices = indices[:n]
    n = len(indices)

    cam_engine = ViTGradCAM(model, block_idx=block_idx)
    device = next(model.parameters()).device

    fig, axes = plt.subplots(n, 3, figsize=(9, 2.7 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image = images[idx].to(device)
        true_class = int(labels[idx])

        cam, pred_class, conf = cam_engine(image, target_class=None)

        # column 1: original
        img_np = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[row, 0].imshow(img_np)
        marker = "OK" if pred_class == true_class else "MISS"
        axes[row, 0].set_title(
            f"[{marker}] idx {idx} | true {true_class + 1} | pred {pred_class + 1} "
            f"(conf {conf:.2f})",
            fontsize=9, loc="left",
        )
        axes[row, 0].axis("off")

        # column 2: raw heatmap
        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title("Grad-CAM heatmap", fontsize=9)
        axes[row, 1].axis("off")

        # column 3: overlay
        axes[row, 2].imshow(overlay_cam(image.cpu(), cam, alpha=0.5))
        axes[row, 2].set_title("Overlay", fontsize=9)
        axes[row, 2].axis("off")

    cam_engine.remove_hooks()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for ViT-Small")
    parser.add_argument("--checkpoint", type=Path, default=Path("./checkpoints/best.pt"),
                        help="Path to trained model checkpoint")
    parser.add_argument("--n", type=int, default=12,
                        help="Number of images to visualize")
    parser.add_argument("--block-idx", type=int, default=-2,
                        help="Transformer block to hook (-2 = before final V-V block)")
    parser.add_argument("--save-path", type=Path, default=Path("./checkpoints/gradcam.png"),
                        help="Where to save the figure")
    parser.add_argument("--vv-blocks", type=int, default=1,
                        help="V-V attention blocks (must match training config)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load val tensors
    _, _, val_images, val_labels = build_tensors()

    # build model and load checkpoint
    model = vit_small(num_classes=NUM_CLASSES, vv_attn_last_n_blocks=args.vv_blocks).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (val_acc {ckpt['val_acc']:.4f})")

    visualize_gradcam_grid(
        model,
        val_images,
        val_labels,
        n=args.n,
        block_idx=args.block_idx,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()