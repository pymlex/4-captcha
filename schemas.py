from pydantic import BaseModel, Field


class SplitCounts(BaseModel):
    train: int = 100_000
    val: int = 5_000
    test: int = 5_000
    adv_train: int = 20_000
    adv_val: int = 1_000
    adv_test: int = 1_000


class ImageSpec(BaseModel):
    width: int = 320
    height: int = 80
    channels: int = 1


class FGSMConfig(BaseModel):
    epsilon_low: float = 0.015
    epsilon_high: float = 0.03


class TrainMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_clean_loss: float | None = None
    val_adv_loss: float | None = None
    val_clean_exact_match: float | None = None
    val_adv_exact_match: float | None = None


class PositionMetrics(BaseModel):
    accuracy: list[float] = Field(min_length=4, max_length=4)
    precision: list[float] = Field(min_length=4, max_length=4)
    recall: list[float] = Field(min_length=4, max_length=4)
    mcc: list[float] = Field(min_length=4, max_length=4)


class EvalResult(BaseModel):
    model_name: str
    checkpoint_stage: str
    split: str
    exact_match: float
    position: PositionMetrics
    robustness_gap: float | None = None
    attack_success_rate: float | None = None


class TestPredictions(BaseModel):
    model_name: str
    checkpoint_stage: str
    split: str
    labels: list[str]
    predictions: list[str]
