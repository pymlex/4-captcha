import numpy as np

from data.labels import all_combinations


def split_combination_pools(
    seed: int = 42,
    train_pool_size: int = 8000,
    val_pool_size: int = 1000,
    test_pool_size: int = 1000,
) -> dict[str, np.ndarray]:
    """
    Partition the 10^4 combinations into disjoint pools for train, val, and test.

    Training images sample with replacement from the train pool only.
    Validation and test images sample from their pools without combo overlap
    with the train pool.
    """
    combos = all_combinations()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(combos))
    shuffled = combos[perm]
    train_pool = shuffled[:train_pool_size]
    val_pool = shuffled[train_pool_size : train_pool_size + val_pool_size]
    test_pool = shuffled[
        train_pool_size + val_pool_size : train_pool_size + val_pool_size + test_pool_size
    ]
    return {"train": train_pool, "val": val_pool, "test": test_pool}


def sample_combos(
    pool: np.ndarray,
    count: int,
    rng: np.random.Generator,
    replace: bool,
) -> np.ndarray:
    indices = rng.choice(len(pool), size=count, replace=replace)
    return pool[indices]
