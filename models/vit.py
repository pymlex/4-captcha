import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int = 80,
        img_width: int = 320,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CaptchaViT(nn.Module):
    """Vision Transformer trained from scratch for captcha recognition."""

    def __init__(
        self,
        img_height: int = 80,
        img_width: int = 320,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        num_positions: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_height, img_width, patch_size, 1, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_queries = nn.Parameter(torch.randn(1, num_positions, embed_dim))
        self.heads = nn.ModuleList(
            [nn.Linear(embed_dim, num_classes) for _ in range(num_positions)]
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pos_feat = self.pos_queries.expand(batch, -1, -1)
        attn_scores = torch.matmul(pos_feat, tokens.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pos_repr = torch.matmul(attn_weights, tokens)
        logits = torch.stack(
            [head(pos_repr[:, i, :]) for i, head in enumerate(self.heads)],
            dim=1,
        )
        return logits
