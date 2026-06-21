import _path
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from config import get_settings
from data.dataset import CaptchaDataset
from train.factory import build_model, default_lr
from train.loop import (
    evaluate_loader,
    make_loader,
    train_one_epoch,
    write_history,
)
from train.scheduler import warmup_cosine_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device
from utils.seed import set_seed


def mixed_loader(
    clean_root: Path,
    adv_root: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ConcatDataset(
        [CaptchaDataset(clean_root), CaptchaDataset(adv_root)]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def finetune(model_name: str) -> None:
    settings = get_settings()
    set_seed(settings.seed)
    device = get_device()
    clean_dir = settings.data_dir / "clean"
    adv_dir = settings.data_dir / "adv" / model_name
    train_loader = mixed_loader(
        clean_dir / "train",
        adv_dir / "train",
        settings.batch_size,
        settings.num_workers_effective,
    )
    val_clean_loader = make_loader(
        clean_dir / "val",
        settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers_effective,
        max_samples=1000,
    )
    val_adv_loader = make_loader(
        adv_dir / "val",
        settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers_effective,
    )
    model = build_model(model_name, settings).to(device)
    ckpt = settings.checkpoint_dir / model_name / "clean" / "final.pt"
    load_checkpoint(ckpt, model)
    lr = default_lr(model_name, settings, finetune=True)
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=settings.weight_decay
    )
    epochs = settings.finetune_epochs_effective
    total_steps = len(train_loader) * epochs
    scheduler = warmup_cosine_scheduler(
        optimizer, settings.warmup_steps, total_steps
    )
    ckpt_dir = settings.checkpoint_dir / model_name / "finetune"
    metrics_path = settings.output_dir / "metrics" / f"{model_name}_finetune.json"
    history: list[dict] = []
    write_history(metrics_path, history)
    record: dict = {}
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
        val_clean = evaluate_loader(
            model,
            val_clean_loader,
            device,
            desc=f"val clean {epoch}/{epochs}",
        )
        val_adv = evaluate_loader(
            model,
            val_adv_loader,
            device,
            desc=f"val adv {epoch}/{epochs}",
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_clean_loss": val_clean["loss"],
            "val_adv_loss": val_adv["loss"],
            "val_clean_exact_match": val_clean["exact_match"],
            "val_adv_exact_match": val_adv["exact_match"],
        }
        history.append(record)
        write_history(metrics_path, history)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_clean_em=f"{val_clean['exact_match']:.4f}",
            val_adv_em=f"{val_adv['exact_match']:.4f}",
        )
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch,
            record,
        )
    save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        optimizer,
        epochs,
        record,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], required=True)
    args = parser.parse_args()
    finetune(args.model)
