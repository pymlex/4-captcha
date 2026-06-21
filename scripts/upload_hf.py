from pathlib import Path

from huggingface_hub import HfApi, create_repo

from config import get_settings


def upload_dataset() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)
    create_repo(
        settings.hf_dataset_repo,
        repo_type="dataset",
        exist_ok=True,
        token=settings.hf_token or None,
    )
    api.upload_folder(
        folder_path=str(settings.data_dir),
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        token=settings.hf_token or None,
    )


def upload_models() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)
    create_repo(
        settings.hf_model_repo,
        repo_type="model",
        exist_ok=True,
        token=settings.hf_token or None,
    )
    api.upload_folder(
        folder_path=str(settings.checkpoint_dir),
        repo_id=settings.hf_model_repo,
        repo_type="model",
        token=settings.hf_token or None,
    )


def upload_outputs() -> None:
    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)
    create_repo(
        settings.hf_model_repo,
        repo_type="model",
        exist_ok=True,
        token=settings.hf_token or None,
    )
    for sub in ("metrics", "plots", "predictions"):
        path = settings.output_dir / sub
        if path.exists():
            api.upload_folder(
                folder_path=str(path),
                path_in_repo=sub,
                repo_id=settings.hf_model_repo,
                repo_type="model",
                token=settings.hf_token or None,
            )


if __name__ == "__main__":
    upload_dataset()
    upload_models()
    upload_outputs()
