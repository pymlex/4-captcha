from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

from schemas import FGSMConfig, ImageSpec, SplitCounts


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = 42
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")

    quick_mode: bool = False
    train_size: int = 100_000
    val_size: int = 5_000
    test_size: int = 5_000
    adv_train_size: int = 20_000
    adv_val_size: int = 1_000
    adv_test_size: int = 1_000

    batch_size: int = 128
    grad_accumulation_steps: int = 1
    num_workers: int = 4

    vit_lr: float = 3e-4
    cnn_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    clean_epochs: int = 20
    finetune_epochs: int = 20

    vit_embed_dim: int = 256
    vit_depth: int = 6
    vit_num_heads: int = 8
    vit_patch_size: int = 16
    vit_mlp_ratio: int = 4

    hf_dataset_repo: str = "pymlex/4-captcha"
    hf_model_repo: str = "pymlex/4-captcha-solvers"
    hf_token: str = ""

    @property
    def image_spec(self) -> ImageSpec:
        return ImageSpec()

    @property
    def fgsm(self) -> FGSMConfig:
        return FGSMConfig()

    @property
    def splits(self) -> SplitCounts:
        if self.quick_mode:
            return SplitCounts(
                train=200,
                val=40,
                test=40,
                adv_train=40,
                adv_val=10,
                adv_test=10,
            )
        return SplitCounts(
            train=self.train_size,
            val=self.val_size,
            test=self.test_size,
            adv_train=self.adv_train_size,
            adv_val=self.adv_val_size,
            adv_test=self.adv_test_size,
        )

    @property
    def clean_epochs_effective(self) -> int:
        return 1 if self.quick_mode else self.clean_epochs

    @property
    def finetune_epochs_effective(self) -> int:
        return 1 if self.quick_mode else self.finetune_epochs

    @property
    def num_workers_effective(self) -> int:
        if os.name == "nt":
            return 0
        return self.num_workers


def get_settings() -> Settings:
    return Settings()
