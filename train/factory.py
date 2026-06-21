import torch.nn as nn

from config import Settings
from models.cnn import CompactCaptchaNet
from models.vit import CaptchaViT


def build_model(name: str, settings: Settings) -> nn.Module:
    if name == "cnn":
        return CompactCaptchaNet()
    if name == "vit":
        spec = settings.image_spec
        return CaptchaViT(
            img_height=spec.height,
            img_width=spec.width,
            patch_size=settings.vit_patch_size,
            embed_dim=settings.vit_embed_dim,
            depth=settings.vit_depth,
            num_heads=settings.vit_num_heads,
            mlp_ratio=settings.vit_mlp_ratio,
        )
    raise ValueError(f"Unknown model: {name}")


def default_lr(name: str, settings: Settings, finetune: bool = False) -> float:
    if finetune:
        return settings.finetune_lr
    return settings.vit_lr if name == "vit" else settings.cnn_lr
