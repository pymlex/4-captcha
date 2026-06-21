import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GITHUB_PATHS = [
    "outputs/metrics",
    "outputs/predictions",
    "outputs/plots",
    "hub/model_card.md",
    "hub/dataset_card.md",
]


def publish_github_outputs(
    message: str = "Publish evaluation artefacts and hub cards",
) -> None:
    for rel in GITHUB_PATHS:
        subprocess.run(["git", "add", rel], cwd=ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No GitHub changes to commit.")
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)
    print("GitHub outputs published.")
