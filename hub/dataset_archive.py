import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from config import Settings

SPLIT_RELPATHS = [
    "clean/train",
    "clean/val",
    "clean/test",
    "adv/vit/train",
    "adv/vit/val",
    "adv/vit/test",
    "adv/cnn/train",
    "adv/cnn/val",
    "adv/cnn/test",
]

META_FILENAME = "data_archive_meta.json"


def iter_dataset_files(data_dir: Path):
    for sub in ("clean", "adv"):
        root = data_dir / sub
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                yield path


def dataset_fingerprint(data_dir: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for path in iter_dataset_files(data_dir):
        file_count += 1
        total_bytes += path.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def meta_path(dist_dir: Path) -> Path:
    return dist_dir / META_FILENAME


def archive_is_current(
    data_dir: Path,
    archive_path: Path,
    dist_dir: Path,
) -> bool:
    meta_file = meta_path(dist_dir)
    if not archive_path.exists() or not meta_file.exists():
        return False
    stored = json.loads(meta_file.read_text(encoding="utf-8"))
    current = dataset_fingerprint(data_dir)
    return stored == current


def build_archive(
    data_dir: Path,
    archive_path: Path,
    dist_dir: Path,
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_dataset_files(data_dir))
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in tqdm(files, desc="build data.tar.gz"):
            arcname = path.relative_to(data_dir).as_posix()
            tar.add(path, arcname=arcname)
    meta_path(dist_dir).write_text(
        json.dumps(dataset_fingerprint(data_dir), indent=2),
        encoding="utf-8",
    )
    size_gb = archive_path.stat().st_size / 1e9
    print(f"Archive written: {archive_path} ({size_gb:.2f} GB)")
    return archive_path


def ensure_archive(settings: Settings, force: bool = False) -> Path:
    data_dir = settings.data_dir
    archive_path = settings.dataset_archive_path
    dist_dir = archive_path.parent
    if force or not archive_is_current(data_dir, archive_path, dist_dir):
        build_archive(data_dir, archive_path, dist_dir)
    else:
        print(f"Reusing archive: {archive_path}")
    return archive_path


def build_preview(
    data_dir: Path,
    preview_dir: Path,
    per_split: int,
) -> Path:
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    for rel in SPLIT_RELPATHS:
        src = data_dir / rel
        if not src.exists():
            continue
        pngs = sorted(src.glob("*.png"))[:per_split]
        if not pngs:
            continue
        dst = preview_dir / rel
        dst.mkdir(parents=True, exist_ok=True)
        labels_df = pd.read_csv(src / "labels.csv", dtype={"label": str})
        lookup = dict(zip(labels_df["filename"], labels_df["label"]))
        rows = []
        for png in pngs:
            shutil.copy2(png, dst / png.name)
            rows.append({"filename": png.name, "label": lookup[png.name]})
        pd.DataFrame(rows).to_csv(dst / "labels.csv", index=False)
    adv_meta = data_dir / "adv"
    if adv_meta.exists():
        for meta in adv_meta.rglob("meta.json"):
            rel = meta.relative_to(data_dir)
            out = preview_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta, out)
    print(f"Preview samples: {preview_dir}")
    return preview_dir


def ensure_preview(settings: Settings) -> Path:
    return build_preview(
        settings.data_dir,
        settings.dataset_preview_dir,
        settings.preview_per_split,
    )
