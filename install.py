import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/pymlex/4-captcha.git"
GITHUBLEX_URL = "https://github.com/pymlex/githublex.git"
PROJECT_MARKERS = ("config.py", "main.py", "requirements.txt")


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def is_project_root(path: Path) -> bool:
    return all((path / name).exists() for name in PROJECT_MARKERS)


def clone_repo(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", REPO_URL, str(target)])


def pull_repo(root: Path) -> None:
    run(["git", "-C", str(root), "pull", "--ff-only"])


def ensure_repository(explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        root = explicit_dir.expanduser().resolve()
        if is_project_root(root):
            pull_repo(root)
            return root
        if (root / ".git").exists():
            return root
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Target directory is not empty: {root}")
        clone_repo(root)
        return root

    cwd = Path.cwd().resolve()
    if is_project_root(cwd):
        pull_repo(cwd)
        return cwd

    target = cwd / "4-captcha"
    if is_project_root(target):
        pull_repo(target)
        return target

    clone_repo(target)
    return target


def ensure_env_file(root: Path) -> None:
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists() or not example_path.exists():
        return
    shutil.copy(example_path, env_path)


def install_dependencies(root: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(root / "requirements.txt"),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"git+{GITHUBLEX_URL}",
        ]
    )


def login_github() -> None:
    from githublex import gh_login, gh_setup

    gh_setup()
    gh_login()


def load_hf_token(root: Path) -> str:
    from dotenv import load_dotenv

    load_dotenv(root / ".env")
    return os.environ.get("HF_TOKEN", "").strip()


def login_huggingface(root: Path) -> None:
    from huggingface_hub import login

    token = load_hf_token(root)
    if token:
        login(token=token)
        return
    login()


def install(explicit_dir: Path | None = None) -> Path:
    root = ensure_repository(explicit_dir)
    os.chdir(root)
    ensure_env_file(root)
    install_dependencies(root)
    login_github()
    login_huggingface(root)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone or update 4-captcha, install deps, login to GitHub and Hugging Face."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Clone or update the repository at this path. Default: cwd or ./4-captcha",
    )
    args = parser.parse_args()
    root = install(args.dir)
    print(f"Installation complete: {root}")


if __name__ == "__main__":
    main()
