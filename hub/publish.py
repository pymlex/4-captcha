from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_large_folder

from config import get_settings
from hub.model_card_builder import build_model_card

ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = ROOT / "hub" / "dataset_card.md"

DATASET_DESCRIPTION = (
    "A synthetic grayscale four-digit captcha dataset with clean splits and "
    "per-model FGSM adversarial images for robust digit-string recognition."
)

MODEL_DESCRIPTION = (
    "ViT and CNN captcha solvers with clean and adversarially fine-tuned "
    "checkpoints, training curves, and exact-match evaluation metrics."
)


def upload_dataset() -> None:
    settings = get_settings()
    token = settings.hf_token or None
    api = HfApi(token=token)
    create_repo(
        settings.hf_dataset_repo,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )
    api.update_repo_settings(
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        description=DATASET_DESCRIPTION,
    )
    print(f"Uploading dataset from {settings.data_dir} with upload_large_folder")
    upload_large_folder(
        repo_id=settings.hf_dataset_repo,
        folder_path=settings.data_dir,
        repo_type="dataset",
        num_workers=8,
        print_report=True,
        print_report_every=30,
    )
    api.upload_file(
        path_or_fileobj=str(DATASET_CARD),
        path_in_repo="README.md",
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        commit_message="Update dataset card",
        token=token,
    )
    print(f"Dataset upload complete: {settings.hf_dataset_repo}")


def upload_models() -> None:
    settings = get_settings()
    token = settings.hf_token or None
    repo = settings.hf_model_repo
    api = HfApi(token=token)
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api.update_repo_settings(
        repo_id=repo,
        repo_type="model",
        description=MODEL_DESCRIPTION,
    )

    plots_dir = settings.output_dir / "plots"
    metrics_dir = settings.output_dir / "metrics"

    if plots_dir.exists():
        print(f"Uploading plots to {repo}")
        api.upload_folder(
            folder_path=str(plots_dir),
            path_in_repo="plots",
            repo_id=repo,
            repo_type="model",
            token=token,
            commit_message="Upload evaluation and training plots",
        )

    if metrics_dir.exists():
        print(f"Uploading metrics to {repo}")
        api.upload_folder(
            folder_path=str(metrics_dir),
            path_in_repo="metrics",
            repo_id=repo,
            repo_type="model",
            token=token,
            commit_message="Upload metrics",
        )

    print(f"Uploading checkpoints to {repo}")
    api.upload_folder(
        folder_path=str(settings.checkpoint_dir),
        path_in_repo="checkpoints",
        repo_id=repo,
        repo_type="model",
        token=token,
        commit_message="Upload checkpoints",
    )

    card = build_model_card(settings.output_dir)
    card_path = ROOT / "hub" / "_model_README.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="Update model card",
        token=token,
    )
    print(f"Model upload complete: {repo}")
