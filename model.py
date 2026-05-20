import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 6

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    dropout: float = 0.1
    attn_dropout: float = 0.0

    # SEM-CLIP-inspired: replace Q-K-V attention with V-V attention; set to 0 to disable
    vv_attn_last_n_blocks: int = 1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ----------------------------------------------------------------------
# Patch embedding
# ----------------------------------------------------------------------
class PatchEmbedding(nn.Module):

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.embed_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ----------------------------------------------------------------------
# Attention (standard Q-K-V and SEM-CLIP-inspired V-V)
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):

    def __init__(self, cfg: ViTConfig, use_vv: bool = False) -> None:
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_vv = use_vv

        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SEM-CLIP modifications: use V for both query and key
        if self.use_vv:
            attn = (v @ v.transpose(-2, -1)) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------------------------------------------------
# MLP and Transformer block
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ViTConfig, use_vv: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = MultiHeadAttention(cfg, use_vv=use_vv)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------------------------------------------------
# Vision Transformer
# ----------------------------------------------------------------------
class VisionTransformer(nn.Module):

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches + 1, cfg.embed_dim)
        )
        self.pos_drop = nn.Dropout(cfg.dropout)

        # build transformer blocks, marking the final N for V-V attention
        n_vv = min(cfg.vv_attn_last_n_blocks, cfg.depth)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, use_vv=(i >= cfg.depth - n_vv))
            for i in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # [CLS] token representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat)


# ----------------------------------------------------------------------
# Define default ViT and smaller ViT
# ----------------------------------------------------------------------
def vit_base(num_classes: int = 6, vv_attn_last_n_blocks: int = 1) -> VisionTransformer:
    #  default configuration from the ViT paper
    return VisionTransformer(ViTConfig(
        num_classes=num_classes,
        vv_attn_last_n_blocks=vv_attn_last_n_blocks,
    ))


def vit_small(num_classes: int = 6, vv_attn_last_n_blocks: int = 1) -> VisionTransformer:
    # smaller configuseful for fast experimentation on small datasets like Carinthia
    return VisionTransformer(ViTConfig(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        vv_attn_last_n_blocks=vv_attn_last_n_blocks,
    ))


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = vit_small(num_classes=6, vv_attn_last_n_blocks=1)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(logits.shape)}")
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")