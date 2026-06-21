import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from config import get_settings
from train.factory import build_model, default_lr
from train.loop import (
    append_metrics,
    evaluate_loader,
    make_loader,
    train_one_epoch,
)
from train.scheduler import warmup_cosine_scheduler
from utils.checkpoint import save_checkpoint
from utils.device import get_device
from utils.seed import set_seed


def train_clean(model_name: str) -> None:
    settings = get_settings()
    set_seed(settings.seed)
    device = get_device()
    data_dir = settings.data_dir / "clean"
    train_loader = make_loader(
        data_dir / "train",
        settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers_effective,
    )
    val_loader = make_loader(
        data_dir / "val",
        settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers_effective,
    )
    model = build_model(model_name, settings).to(device)
    lr = default_lr(model_name, settings)
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=settings.weight_decay
    )
    total_steps = len(train_loader) * settings.clean_epochs_effective
    scheduler = warmup_cosine_scheduler(
        optimizer, settings.warmup_steps, total_steps
    )
    ckpt_dir = settings.checkpoint_dir / model_name / "clean"
    metrics_path = settings.output_dir / "metrics" / f"{model_name}_clean.json"
    best_exact = -1.0
    best_epoch = 0
    for epoch in range(1, settings.clean_epochs_effective + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            settings.grad_accumulation_steps,
        )
        val_metrics = evaluate_loader(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_clean_loss": val_metrics["loss"],
            "val_clean_exact_match": val_metrics["exact_match"],
        }
        append_metrics(metrics_path, record)
        if val_metrics["exact_match"] >= best_exact:
            best_exact = val_metrics["exact_match"]
            best_epoch = epoch
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
            )
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
        )
    save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        optimizer,
        settings.clean_epochs_effective,
        {"best_epoch": best_epoch, "best_exact_match": best_exact},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], required=True)
    args = parser.parse_args()
    train_clean(args.model)
