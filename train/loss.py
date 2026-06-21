import torch
import torch.nn as nn


def captcha_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Sum cross-entropy over four digit positions.

    logits: (batch, 4, 10), targets: (batch, 4)
    """
    criterion = nn.CrossEntropyLoss()
    losses = [criterion(logits[:, pos, :], targets[:, pos]) for pos in range(4)]
    return torch.stack(losses).sum()


def exact_match(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    matches = (preds == targets).all(dim=1).float().mean().item()
    return matches
