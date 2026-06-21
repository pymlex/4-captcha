import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.dataset import CaptchaDataset
from train.loss import captcha_loss, exact_match
from utils.device import get_device


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


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    batches = 0
    for images, targets in loader:
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
) -> float:
    model.train()
    running = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step_idx, (images, targets) in enumerate(
        tqdm(loader, desc="train", leave=False)
    ):
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
        running += loss.item() * grad_accumulation_steps
        steps += 1
    return running / max(1, steps)


def append_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
    history.append(record)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
