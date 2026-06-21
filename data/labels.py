import numpy as np
import torch


DIGITS = np.arange(10, dtype=np.int64)


def all_combinations() -> np.ndarray:
    """Return all 10^4 four-digit strings as an array of shape (10000, 4)."""
    combos = np.indices((10, 10, 10, 10)).reshape(4, -1).T
    return combos.astype(np.int64)


def combo_to_string(combo: np.ndarray) -> str:
    return "".join(str(d) for d in combo)


def string_to_targets(label: str) -> torch.Tensor:
    return torch.tensor([int(ch) for ch in label], dtype=torch.long)


def strings_to_targets(labels: list[str]) -> torch.Tensor:
    arr = np.array([[int(ch) for ch in label] for label in labels], dtype=np.int64)
    return torch.from_numpy(arr)


def targets_to_strings(targets: torch.Tensor) -> list[str]:
    return ["".join(str(d.item()) for d in row) for row in targets]
