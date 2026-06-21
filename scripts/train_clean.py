import _path
import argparse

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from config import get_settings
from train.factory import build_model, default_lr
from train.loop import (
    evaluate_loader,
    make_loader,
    train_one_epoch,
    write_history,
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
    epochs = settings.clean_epochs_effective
    total_steps = len(train_loader) * epochs
    scheduler = warmup_cosine_scheduler(
        optimizer, settings.warmup_steps, total_steps
    )
    ckpt_dir = settings.checkpoint_dir / model_name / "clean"
    metrics_path = settings.output_dir / "metrics" / f"{model_name}_clean.json"
    history: list[dict] = []
    write_history(metrics_path, history)
    best_exact = -1.0
    best_epoch = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            settings.grad_accumulation_steps,
            desc=f"train {epoch}/{epochs}",
        )
        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            desc=f"val {epoch}/{epochs}",
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_clean_loss": val_metrics["loss"],
            "val_clean_exact_match": val_metrics["exact_match"],
        }
        history.append(record)
        write_history(metrics_path, history)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_em=f"{val_metrics['exact_match']:.4f}",
        )
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
        epochs,
        {"best_epoch": best_epoch, "best_exact_match": best_exact},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], required=True)
    args = parser.parse_args()
    train_clean(args.model)
