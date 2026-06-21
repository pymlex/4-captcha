from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptchaDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        labels_path = self.root / "labels.csv"
        self.df = pd.read_csv(labels_path, dtype={"label": str})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        path = self.root / row["filename"]
        image = Image.open(path).convert("L")
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).contiguous()
        label = str(row["label"]).zfill(4)
        targets = torch.tensor(
            [int(ch) for ch in label], dtype=torch.long
        )
        return tensor, targets
