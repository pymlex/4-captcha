import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from attacks.fgsm import fgsm_attack
from config import get_settings
from data.dataset import CaptchaDataset
from train.factory import build_model
from utils.checkpoint import load_checkpoint
from utils.device import get_device
from utils.seed import set_seed


def epsilon_schedule(count: int, seed: int) -> np.ndarray:
    half = count // 2
    eps = np.array(
        [0.015] * half + [0.03] * (count - half),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    return eps


def adv_split_dir(data_dir: Path, model_name: str, split: str) -> Path:
    return data_dir / "adv" / model_name / split


def generate_split(
    model: torch.nn.Module,
    source_root: Path,
    out_root: Path,
    indices: np.ndarray,
    epsilons: np.ndarray,
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = CaptchaDataset(source_root)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    records = []
    offset = 0
    for images, targets in tqdm(loader, desc=f"fgsm {split_name}"):
        images = images.to(device)
        targets = targets.to(device)
        batch_eps = epsilons[offset : offset + len(images)]
        adv_batch = []
        for i in range(len(images)):
            adv = fgsm_attack(
                model,
                images[i : i + 1],
                targets[i : i + 1],
                float(batch_eps[i]),
            )
            adv_batch.append(adv.cpu())
        adv_images = torch.cat(adv_batch, dim=0)
        for j in range(adv_images.shape[0]):
            arr = (adv_images[j, 0].numpy() * 255.0).astype(np.uint8)
            filename = f"{split_name}_{offset + j:06d}.png"
            Image.fromarray(arr).save(out_root / filename)
            label = "".join(str(d.item()) for d in targets[j])
            records.append({"filename": filename, "label": label, "epsilon": float(batch_eps[j])})
        offset += len(images)
    pd.DataFrame(records).to_csv(out_root / "labels.csv", index=False)


def sample_indices(
    total: int,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=count, replace=False)


def generate_fgsm(model_name: str) -> None:
    settings = get_settings()
    set_seed(settings.seed)
    device = get_device()
    model = build_model(model_name, settings).to(device)
    ckpt = settings.checkpoint_dir / model_name / "clean" / "final.pt"
    load_checkpoint(ckpt, model)
    splits = settings.splits
    clean_dir = settings.data_dir / "clean"
    train_ds = CaptchaDataset(clean_dir / "train")
    val_ds = CaptchaDataset(clean_dir / "val")
    test_ds = CaptchaDataset(clean_dir / "test")

    train_idx = sample_indices(len(train_ds), splits.adv_train, settings.seed)
    val_idx = sample_indices(len(val_ds), splits.adv_val, settings.seed + 1)
    test_idx = sample_indices(len(test_ds), splits.adv_test, settings.seed + 2)

    train_eps = epsilon_schedule(splits.adv_train, settings.seed)
    val_eps = epsilon_schedule(splits.adv_val, settings.seed + 3)
    test_eps = epsilon_schedule(splits.adv_test, settings.seed + 4)

    generate_split(
        model,
        clean_dir / "train",
        adv_split_dir(settings.data_dir, model_name, "train"),
        train_idx,
        train_eps,
        device,
        settings.batch_size,
        "adv_train",
    )
    generate_split(
        model,
        clean_dir / "val",
        adv_split_dir(settings.data_dir, model_name, "val"),
        val_idx,
        val_eps,
        device,
        settings.batch_size,
        "adv_val",
    )
    generate_split(
        model,
        clean_dir / "test",
        adv_split_dir(settings.data_dir, model_name, "test"),
        test_idx,
        test_eps,
        device,
        settings.batch_size,
        "adv_test",
    )

    meta = {
        "model": model_name,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    meta_path = adv_split_dir(settings.data_dir, model_name, "meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], required=True)
    args = parser.parse_args()
    generate_fgsm(args.model)
