import os
import shutil
import subprocess
import sys
from pathlib import Path

GITHUBLEX_URL = "https://github.com/pymlex/githublex.git"
ROOT = Path(__file__).resolve().parent


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def pull_repository() -> None:
    run(["git", "pull", "--ff-only"], cwd=ROOT)


def ensure_env_file() -> None:
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if env_path.exists() or not example_path.exists():
        return
    shutil.copy(example_path, env_path)


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements.txt"),
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


def load_hf_token() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get("HF_TOKEN", "").strip()


def login_huggingface() -> None:
    from huggingface_hub import login

    token = load_hf_token()
    if token:
        login(token=token)
        return
    login()


def install() -> None:
    pull_repository()
    ensure_env_file()
    install_dependencies()
    login_github()
    login_huggingface()


if __name__ == "__main__":
    install()
