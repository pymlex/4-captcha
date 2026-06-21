import _path
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from config import get_settings
from data.augment import global_augment
from data.combinations import sample_combos, split_combination_pools
from data.labels import combo_to_string
from data.render_digits import render_digits
from data.fonts import load_fonts
from utils.seed import set_seed


def split_dir(data_dir: Path, split: str) -> Path:
    return data_dir / "clean" / split


def reset_clean_data(data_dir: Path) -> None:
    clean_root = data_dir / "clean"
    if clean_root.exists():
        shutil.rmtree(clean_root)


def save_split(
    combos: np.ndarray,
    out_dir: Path,
    width: int,
    height: int,
    rng: np.random.Generator,
    fonts,
    split_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in tqdm(range(len(combos)), desc=f"generate {split_name}"):
        combo = combos[idx]
        base = render_digits(combo, width, height, rng, fonts)
        image = global_augment(base, rng)
        filename = f"{split_name}_{idx:06d}.png"
        Image.fromarray(image).save(out_dir / filename)
        records.append({"filename": filename, "label": combo_to_string(combo).zfill(4)})
    pd.DataFrame(records).to_csv(out_dir / "labels.csv", index=False)


def generate_clean_dataset() -> None:
    settings = get_settings()
    set_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    pools = split_combination_pools(settings.seed)
    width = settings.image_spec.width
    height = settings.image_spec.height
    fonts = load_fonts(height)
    splits = settings.splits

    reset_clean_data(settings.data_dir)

    train_combos = sample_combos(
        pools["train"], splits.train, rng, replace=True
    )
    val_combos = sample_combos(
        pools["val"], splits.val, rng, replace=True
    )
    test_combos = sample_combos(
        pools["test"], splits.test, rng, replace=True
    )

    save_split(
        train_combos,
        split_dir(settings.data_dir, "train"),
        width,
        height,
        rng,
        fonts,
        "train",
    )
    save_split(
        val_combos,
        split_dir(settings.data_dir, "val"),
        width,
        height,
        rng,
        fonts,
        "val",
    )
    save_split(
        test_combos,
        split_dir(settings.data_dir, "test"),
        width,
        height,
        rng,
        fonts,
        "test",
    )

    meta = {
        "seed": settings.seed,
        "train_pool_size": len(pools["train"]),
        "val_pool_size": len(pools["val"]),
        "test_pool_size": len(pools["test"]),
        "counts": splits.model_dump(),
    }
    meta_path = settings.data_dir / "clean" / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate_clean_dataset()
