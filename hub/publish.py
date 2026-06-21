from pathlib import Path

from huggingface_hub import HfApi

from config import get_settings
from hub.dataset_archive import ensure_archive, ensure_preview
from hub.dataset_git_upload import upload_dataset_git
from hub.github_publish import publish_github_outputs
from hub.hf_retry import call_with_retry, ensure_repo

ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = ROOT / "hub" / "dataset_card.md"
MODEL_CARD = ROOT / "hub" / "model_card.md"


def upload_dataset(force_archive: bool = False) -> None:
    settings = get_settings()
    token = settings.hf_token or None
    api = HfApi(token=token)
    ensure_repo(api, settings.hf_dataset_repo, "dataset", token)

    archive_path = ensure_archive(settings, force=force_archive)
    preview_dir = ensure_preview(settings)

    upload_dataset_git(
        settings,
        archive_path,
        preview_dir,
        DATASET_CARD,
    )


def upload_models() -> None:
    settings = get_settings()
    token = settings.hf_token or None
    repo = settings.hf_model_repo
    api = HfApi(token=token)
    ensure_repo(api, repo, "model", token)

    plots_dir = settings.output_dir / "plots"
    metrics_dir = settings.output_dir / "metrics"

    if plots_dir.exists():
        print(f"Uploading plots to {repo}")
        call_with_retry(
            lambda: api.upload_folder(
                folder_path=str(plots_dir),
                path_in_repo="plots",
                repo_id=repo,
                repo_type="model",
                token=token,
                commit_message="Upload evaluation and training plots",
            ),
            "upload plots",
        )

    if metrics_dir.exists():
        print(f"Uploading metrics to {repo}")
        call_with_retry(
            lambda: api.upload_folder(
                folder_path=str(metrics_dir),
                path_in_repo="metrics",
                repo_id=repo,
                repo_type="model",
                token=token,
                commit_message="Upload metrics",
            ),
            "upload metrics",
        )

    print(f"Uploading checkpoints to {repo}")
    call_with_retry(
        lambda: api.upload_folder(
            folder_path=str(settings.checkpoint_dir),
            path_in_repo="checkpoints",
            repo_id=repo,
            repo_type="model",
            token=token,
            commit_message="Upload checkpoints",
        ),
        "upload checkpoints",
    )

    call_with_retry(
        lambda: api.upload_file(
            path_or_fileobj=str(MODEL_CARD),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="Update model card",
            token=token,
        ),
        "upload model card",
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
