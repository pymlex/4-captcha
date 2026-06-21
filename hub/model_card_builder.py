import json
from pathlib import Path


def format_test_table(test_results: list[dict]) -> str:
    lines = [
        "| Model | Stage | Split | Exact match |",
        "|-------|-------|-------|-------------|",
    ]
    for row in test_results:
        em = row["exact_match"]
        lines.append(
            f"| {row['model_name']} | {row['checkpoint_stage']} "
            f"| {row['split']} | {em:.4f} |"
        )
    return "\n".join(lines)


def plot_block(title: str, paths: list[Path], output_dir: Path) -> str:
    blocks = [f"### {title}", ""]
    for path in paths:
        if path.exists():
            rel = path.relative_to(output_dir).as_posix()
            blocks.append(f"![{path.stem}]({rel})")
            blocks.append("")
    return "\n".join(blocks)


def build_model_card(output_dir: Path) -> str:
    test_path = output_dir / "metrics" / "test_results.json"
    test_results = []
    if test_path.exists():
        test_results = json.loads(test_path.read_text(encoding="utf-8"))

    plots = output_dir / "plots"
    sections = [
        "---",
        "license: gpl-3.0",
        "language:",
        "- en",
        "tags:",
        "- captcha",
        "- vision",
        "- adversarial",
        "library_name: pytorch",
        "---",
        "",
        "# 4-captcha solvers",
        "",
        "Checkpoints for four-digit captcha recognition on the "
        "[pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha) "
        "dataset: CompactCaptchaNet and CaptchaViT, each in a clean-trained and "
        "adversarially fine-tuned variant.",
        "",
        "## Task",
        "",
        "Input is a grayscale $320 \\times 80$ image with a four-digit string. "
        "Each model outputs logits $Z \\in \\mathbb{R}^{4 \\times 10}$. "
        "The training loss is",
        "",
        "$$L = \\sum_{p=1}^{4} CE(Z_{:,p,:}, y_p)$$",
        "",
        "FGSM fine-tuning uses",
        "",
        "$$x_{adv} = clip(x + \\varepsilon \\cdot sign(\\nabla_x L), 0, 1)$$",
        "",
        "with $\\varepsilon \\in \\{0.015, 0.03\\}$.",
        "",
        "## Architectures",
        "",
        "**CompactCaptchaNet** — four stride-2 conv blocks, reshape to $(1280, 20)$, "
        "`Conv1d` temporal mixing, adaptive pool to four positions, linear heads. "
        "About 1.4M parameters.",
        "",
        "**CaptchaViT** — patch size $16 \\times 16$, embed dim 256, depth 6, "
        "eight heads, learned position queries over patch tokens. About 4.8M parameters.",
        "",
        "Clean training: 20 epochs. Adversarial fine-tuning: 20 epochs on a "
        "120k mixed set.",
        "",
        "## Checkpoints",
        "",
        "```",
        "checkpoints/",
        "├── vit/",
        "│   ├── clean/",
        "│   └── finetune/",
        "└── cnn/",
        "    ├── clean/",
        "    └── finetune/",
        "```",
        "",
        "Each stage stores `best.pt`, `last.pt`, and `final.pt`.",
        "",
        "## Test exact match",
        "",
    ]

    if test_results:
        sections.append(format_test_table(test_results))
    else:
        sections.append("Run evaluation before upload to populate this table.")

    confusion_paths = (
        sorted((plots / "confusion").glob("*_confusion.png"))
        if (plots / "confusion").exists()
        else []
    )

    sections.extend(
        [
            "",
            "## Training curves",
            "",
            plot_block(
                "ViT clean",
                [
                    plots / "vit_clean_loss.png",
                    plots / "vit_clean_val_metrics.png",
                ],
                output_dir,
            ),
            plot_block(
                "ViT fine-tune",
                [
                    plots / "vit_finetune_loss.png",
                    plots / "vit_finetune_val_metrics.png",
                ],
                output_dir,
            ),
            plot_block(
                "CNN clean",
                [
                    plots / "cnn_clean_loss.png",
                    plots / "cnn_clean_val_metrics.png",
                ],
                output_dir,
            ),
            plot_block(
                "CNN fine-tune",
                [
                    plots / "cnn_finetune_loss.png",
                    plots / "cnn_finetune_val_metrics.png",
                ],
                output_dir,
            ),
            plot_block(
                "Test comparison",
                [plots / "test_model_comparison.png"],
                output_dir,
            ),
            "## Evaluation plots",
            "",
            plot_block("Confusion matrices", confusion_paths, output_dir),
            "## Dataset",
            "",
            "[pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha)",
            "",
            "## Source",
            "",
            "[github.com/pymlex/4-captcha](https://github.com/pymlex/4-captcha)",
            "",
            "## Citation",
            "",
            "```bibtex",
            "@misc{zyukov2026_4captcha,",
            "  author       = {Alex Zyukov},",
            "  title        = {4-captcha: Synthetic Captcha Recognition and Adversarial Fine-tuning},",
            "  year         = {2026},",
            '  howpublished = {\\url{https://github.com/pymlex/4-captcha}}',
            "}",
            "```",
            "",
            "The models are under GPL-3.0 license.",
            "",
        ]
    )
    return "\n".join(sections)
