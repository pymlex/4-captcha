import torch
import torch.nn as nn


class CompactCaptchaNet(nn.Module):
    """Compact CNN for 320x80 grayscale captcha images."""

    def __init__(self, num_positions: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(num_positions),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(256, num_classes) for _ in range(num_positions)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 4, 10)."""
        feat = self.features(x)
        batch = feat.shape[0]
        seq = feat.reshape(batch, 256 * 5, 20)
        pooled = self.temporal(seq)
        pooled = pooled.permute(0, 2, 1)
        logits = torch.stack(
            [head(pooled[:, pos, :]) for pos, head in enumerate(self.heads)],
            dim=1,
        )
        return logits
