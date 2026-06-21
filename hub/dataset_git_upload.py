import os
import shutil
import subprocess
from pathlib import Path

from config import Settings

ARCHIVE_NAME = "data.tar.gz"


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def git_env() -> dict[str, str]:
    return {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}


def ensure_git_lfs() -> None:
    run(["git", "lfs", "version"], cwd=Path.cwd())
    run(["git", "lfs", "install"], cwd=Path.cwd())


def repo_workdir(settings: Settings) -> Path:
    return settings.dataset_archive_path.parent / "hf_dataset_repo"


def clone_url(settings: Settings) -> str:
    token = settings.hf_token
    return (
        f"https://oauth2:{token}@huggingface.co/datasets/"
        f"{settings.hf_dataset_repo}"
    )


def sync_preview(preview_dir: Path, workdir: Path) -> None:
    target = workdir / "preview"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(preview_dir, target)


def sync_gitattributes(workdir: Path) -> None:
    path = workdir / ".gitattributes"
    line = "*.tar.gz filter=lfs diff=lfs merge=lfs -text\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if line not in text:
            path.write_text(text + line, encoding="utf-8")
    else:
        path.write_text(line, encoding="utf-8")


def upload_dataset_git(
    settings: Settings,
    archive_path: Path,
    preview_dir: Path,
    card_path: Path,
) -> None:
    ensure_git_lfs()
    workdir = repo_workdir(settings)
    parent = workdir.parent
    parent.mkdir(parents=True, exist_ok=True)

    if workdir.exists():
        shutil.rmtree(workdir)

    print(f"Cloning {settings.hf_dataset_repo}")
    run(
        ["git", "clone", clone_url(settings), str(workdir.name)],
        cwd=parent,
        env=git_env(),
    )

    shutil.copy2(archive_path, workdir / ARCHIVE_NAME)
    shutil.copy2(card_path, workdir / "README.md")
    sync_preview(preview_dir, workdir)
    sync_gitattributes(workdir)

    run(["git", "lfs", "install", "--local"], cwd=workdir)
    run(["git", "lfs", "track", "*.tar.gz"], cwd=workdir)
    run(
        ["git", "add", ".gitattributes", ARCHIVE_NAME, "README.md", "preview"],
        cwd=workdir,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("Dataset repo on Hugging Face is already up to date.")
        return

    run(
        ["git", "commit", "-m", "Upload dataset archive, preview, and card"],
        cwd=workdir,
    )
    print(f"Pushing {ARCHIVE_NAME} via Git LFS to {settings.hf_dataset_repo}")
    run(["git", "push"], cwd=workdir, env=git_env())
    print(f"Dataset upload complete: {settings.hf_dataset_repo}")
