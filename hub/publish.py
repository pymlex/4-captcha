from pathlib import Path

from huggingface_hub import HfApi, create_repo

from config import get_settings
from hub.model_card_builder import build_model_card

ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = ROOT / "hub" / "dataset_card.md"


def upload_dataset() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)
    token = settings.hf_token or None
    create_repo(
        settings.hf_dataset_repo,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        folder_path=str(settings.data_dir),
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(DATASET_CARD),
        path_in_repo="README.md",
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        token=token,
    )


def upload_models() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)
    token = settings.hf_token or None
    repo = settings.hf_model_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)

    card = build_model_card(settings.output_dir)
    card_path = ROOT / "hub" / "_model_README.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        token=token,
    )

    api.upload_folder(
        folder_path=str(settings.checkpoint_dir),
        path_in_repo="checkpoints",
        repo_id=repo,
        repo_type="model",
        token=token,
    )

    metrics_dir = settings.output_dir / "metrics"
    if metrics_dir.exists():
        api.upload_folder(
            folder_path=str(metrics_dir),
            path_in_repo="metrics",
            repo_id=repo,
            repo_type="model",
            token=token,
        )

    plots_dir = settings.output_dir / "plots"
    if plots_dir.exists():
        api.upload_folder(
            folder_path=str(plots_dir),
            path_in_repo="plots",
            repo_id=repo,
            repo_type="model",
            token=token,
        )
