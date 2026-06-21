import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from schemas import EvalResult


def load_test_results(path: Path) -> list[EvalResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalResult.model_validate(item) for item in raw]


def test_comparison_chart(
    results: list[EvalResult],
    out_path: Path,
) -> None:
    model_order = [
        ("vit", "clean"),
        ("vit", "finetune"),
        ("cnn", "clean"),
        ("cnn", "finetune"),
    ]
    model_labels = ["ViT", "ViT-FT", "CNN", "CNN-FT"]
    metric_groups = [
        ("Clean EM", "clean_test", "exact_match"),
        ("Adv EM", "adv_test", "exact_match"),
        ("Robustness gap", "adv_test", "robustness_gap"),
        ("Attack success", "adv_test", "attack_success_rate"),
    ]
    lookup = {
        (r.model_name, r.checkpoint_stage, r.split): r for r in results
    }
    n_groups = len(metric_groups)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_groups)
    width = 0.18
    for m_idx, (name, stage) in enumerate(model_order):
        values = []
        for _, split, field in metric_groups:
            key = (name, stage, split)
            res = lookup[key]
            values.append(getattr(res, field))
        offset = (m_idx - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=model_labels[m_idx],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in metric_groups])
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def training_curves(
    metrics_path: Path,
    out_dir: Path,
    title_prefix: str,
) -> None:
    if not metrics_path.exists():
        return
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="train loss")
    if "val_clean_loss" in history[0]:
        ax.plot(
            epochs,
            [r["val_clean_loss"] for r in history],
            marker="s",
            label="val clean loss",
        )
    if "val_adv_loss" in history[0]:
        ax.plot(
            epochs,
            [r["val_adv_loss"] for r in history],
            marker="^",
            label="val adv loss",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} — loss")
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{title_prefix}_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(epochs, train_loss, marker="o", label="train loss")
    if "val_clean_loss" in history[0]:
        ax.loglog(
            epochs,
            [r["val_clean_loss"] for r in history],
            marker="s",
            label="val clean loss",
        )
    if "val_adv_loss" in history[0]:
        ax.loglog(
            epochs,
            [r["val_adv_loss"] for r in history],
            marker="^",
            label="val adv loss",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} — chinchilla loss")
    ax.legend()
    ax.grid(alpha=0.5, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{title_prefix}_loss_chinchilla.png", dpi=150)
    plt.close(fig)

    if "val_clean_exact_match" in history[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epochs,
            [r["val_clean_exact_match"] for r in history],
            marker="s",
            label="val clean EM",
        )
        if "val_adv_exact_match" in history[0]:
            ax.plot(
                epochs,
                [r["val_adv_exact_match"] for r in history],
                marker="^",
                label="val adv EM",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Exact match")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{title_prefix} — validation metrics")
        ax.legend()
        ax.grid(alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{title_prefix}_val_metrics.png", dpi=150)
        plt.close(fig)


def plot_confusion(
    cm_path: Path,
    out_dir: Path,
    title: str,
) -> None:
    data = np.load(cm_path)
    if "confusion" in data:
        cm = data["confusion"]
    else:
        cm = sum(data[f"pos_{i}"] for i in range(4))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        ax=ax,
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = title.replace(" ", "_")
    fig.savefig(out_dir / f"{safe}_confusion.png", dpi=150)
    plt.close(fig)


def generate_all_plots(output_dir: Path) -> None:
    results_path = output_dir / "metrics" / "test_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}. Run python scripts/evaluate.py first."
        )
    results = load_test_results(results_path)
    plot_dir = output_dir / "plots"
    test_comparison_chart(results, plot_dir / "test_model_comparison.png")
    for model in ("vit", "cnn"):
        training_curves(
            output_dir / "metrics" / f"{model}_clean.json",
            plot_dir,
            f"{model}_clean",
        )
        training_curves(
            output_dir / "metrics" / f"{model}_finetune.json",
            plot_dir,
            f"{model}_finetune",
        )
    cm_dir = output_dir / "confusion"
    if not cm_dir.exists() or not any(cm_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No confusion matrices in {cm_dir}. Run python scripts/evaluate.py first."
        )
    for cm_file in cm_dir.glob("*.npz"):
        plot_confusion(
            cm_file,
            plot_dir / "confusion",
            cm_file.stem,
        )
