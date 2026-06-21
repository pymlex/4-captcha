from pathlib import Path

from huggingface_hub import HfApi, create_repo

from config import get_settings
from hub.dataset_archive import ensure_archive, ensure_preview
from hub.github_publish import publish_github_outputs

ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = ROOT / "hub" / "dataset_card.md"
MODEL_CARD = ROOT / "hub" / "model_card.md"


def upload_dataset(force_archive: bool = False) -> None:
    settings = get_settings()
    token = settings.hf_token or None
    api = HfApi(token=token)
    create_repo(
        settings.hf_dataset_repo,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )

    archive_path = ensure_archive(settings, force=force_archive)
    preview_dir = ensure_preview(settings)

    print(f"Uploading {archive_path.name} to {settings.hf_dataset_repo}")
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=archive_path.name,
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        commit_message="Upload dataset archive",
        token=token,
    )

    print(f"Uploading preview samples to {settings.hf_dataset_repo}")
    api.upload_folder(
        folder_path=str(preview_dir),
        path_in_repo="preview",
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        token=token,
        commit_message="Upload preview samples",
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

    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="Update model card",
        token=token,
    )
    print(f"Model upload complete: {repo}")


def publish_all(
    github: bool = True,
    hf_dataset: bool = True,
    hf_models: bool = True,
    force_archive: bool = False,
) -> None:
    if github:
        publish_github_outputs()
    if hf_dataset:
        upload_dataset(force_archive=force_archive)
    if hf_models:
        upload_models()
