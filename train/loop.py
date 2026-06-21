import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.dataset import CaptchaDataset
from train.loss import captcha_loss, exact_match


def make_loader(
    root: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int | None = None,
) -> DataLoader:
    dataset = CaptchaDataset(root)
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    batches = 0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = captcha_loss(logits, targets)
        total_loss += loss.item()
        total_exact += exact_match(logits, targets)
        batches += 1
    return {
        "loss": total_loss / batches,
        "exact_match": total_exact / batches,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    grad_accumulation_steps: int,
    desc: str = "train",
) -> float:
    model.train()
    running = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    batch_bar = tqdm(loader, desc=desc, leave=False)
    for step_idx, (images, targets) in enumerate(batch_bar):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = captcha_loss(logits, targets) / grad_accumulation_steps
        loss.backward()
        if (step_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        step_loss = loss.item() * grad_accumulation_steps
        running += step_loss
        steps += 1
        batch_bar.set_postfix(loss=f"{step_loss:.4f}")
    return running / max(1, steps)
