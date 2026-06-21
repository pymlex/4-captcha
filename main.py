import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    if args:
        cmd.extend(args)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="4-captcha ML pipeline")
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "generate",
            "train_clean",
            "fgsm",
            "finetune",
            "evaluate",
            "plot",
            "upload",
        ],
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=["vit", "cnn", "both"],
        default="both",
    )
    args = parser.parse_args()
    models = ["vit", "cnn"] if args.model == "both" else [args.model]

    if args.step in ("all", "generate"):
        run_script("generate_dataset.py")

    if args.step in ("all", "train_clean"):
        for model in models:
            run_script("train_clean.py", ["--model", model])

    if args.step in ("all", "fgsm"):
        for model in models:
            run_script("generate_fgsm.py", ["--model", model])

    if args.step in ("all", "finetune"):
        for model in models:
            run_script("finetune.py", ["--model", model])

    if args.step in ("all", "evaluate"):
        run_script("evaluate.py")

    if args.step in ("all", "plot"):
        run_script("plot_results.py")

    if args.step in ("all", "upload"):
        run_script("publish.py")


if __name__ == "__main__":
    main()
