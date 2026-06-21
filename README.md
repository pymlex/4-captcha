# 4-captcha

Synthetic four-digit captcha recognition with Vision Transformer and compact CNN solvers, FGSM adversarial training, and per-position evaluation.

## Overview

The pipeline generates grayscale $320 \times 80$ captcha images with per-digit font variation, elastic deformation, Bézier noise curves, and FGSM perturbations. Two classifiers output logits $Z \in \mathbb{R}^{B \times 4 \times 10}$. The training objective is

$$L = \sum_{p=1}^{4} CE(Z_{:,p,:}, y_p)$$

FGSM examples follow

$$x_{adv} = clip(x + \varepsilon \cdot sign(\nabla_x L), 0, 1)$$

with $\varepsilon \in \{0.015, 0.03\}$.

| Split | Images |
|-------|--------|
| Clean train | 100,000 |
| Clean val | 5,000 |
| Clean test | 5,000 |
| Adv train (per model) | 20,000 |
| Adv val (per model) | 1,000 |
| Adv test (per model) | 1,000 |

## Project tree

```
4-captcha/
├── main.py
├── config.py
├── schemas.py
├── requirements.txt
├── .env.example
├── data/
│   ├── augment.py
│   ├── combinations.py
│   ├── dataset.py
│   ├── fonts.py
│   ├── labels.py
│   └── render_digits.py
├── models/
│   ├── cnn.py          # CompactCaptchaNet (~1.4M params)
│   └── vit.py          # CaptchaViT (~4.8M params)
├── train/
│   ├── factory.py
│   ├── loop.py
│   ├── loss.py
│   └── scheduler.py
├── attacks/
│   └── fgsm.py
├── eval/
│   ├── metrics.py
│   └── plots.py
├── scripts/
│   ├── generate_dataset.py
│   ├── train_clean.py
│   ├── generate_fgsm.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── upload_hf.py
└── outputs/
    ├── metrics/
    ├── plots/
    └── predictions/
```

## Pipeline

```mermaid
flowchart TB
  subgraph generation [Dataset]
    G[Sample combos, render digits, global augment, PNG + labels]
  end
  subgraph train [Training]
    T[Clean train ViT/CNN, FGSM per model, fine-tune mixed 120k]
  end
  subgraph eval [Evaluation]
    E[Clean test 5k, Adv test 1k, metrics + plots]
  end
  G --> T --> E
```

## Setup

Copy environment defaults and install dependencies. Full-run settings: 132,000 images, CLEAN_EPOCHS=50, FINETUNE_EPOCHS=1, QUICK_MODE=false. Set QUICK_MODE=true only for local smoke tests.

```bash
cp .env.example .env
pip install -r requirements.txt
```

## Dataset generation

Build clean train, val, and test splits as PNG files with labels.csv.

```bash
python scripts/generate_dataset.py
```

## Clean training

Train ViT or CNN from scratch on clean data. Checkpoints go to checkpoints/{model}/clean/.

```bash
python scripts/train_clean.py --model vit
python scripts/train_clean.py --model cnn
```

## FGSM generation

Generate adversarial splits from a trained clean model. One FGSM set per model.

```bash
python scripts/generate_fgsm.py --model vit
python scripts/generate_fgsm.py --model cnn
```

## Fine-tuning

Fine-tune each model on the mixed clean and adversarial training set.

```bash
python scripts/finetune.py --model vit
python scripts/finetune.py --model cnn
```

## Evaluation

Compute test metrics and save predictions to outputs/.

```bash
python scripts/evaluate.py
```

## Plotting

Render bar charts, training curves, and confusion matrices from outputs/.

```bash
python scripts/plot_results.py
```

## Upload

Push the dataset and checkpoints to Hugging Face. Set HF_TOKEN in .env first.

```bash
python scripts/upload_hf.py
```

## Full pipeline

Run all stages via main.py. Use --step to run a single stage and --model to select vit, cnn, or both.

```bash
python main.py --step all
python main.py --step generate
python main.py --step train_clean --model vit
python main.py --step fgsm --model vit
python main.py --step finetune --model vit
python main.py --step evaluate
python main.py --step plot
python main.py --step upload
```

## Models

**CompactCaptchaNet** — four stride-2 conv blocks, reshape to $(1280, 20)$, `Conv1d` temporal mixing, adaptive pool to four positions, linear heads.

**CaptchaViT** — patch size $16 \times 16$, embed dim 256, depth 6, eight heads, learned position queries attend over patch tokens.

## Hugging Face

| Artefact | Repository |
|----------|------------|
| Dataset | [pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha) |
| Checkpoints, metrics, plots | [pymlex/4-captcha-solvers](https://huggingface.co/pymlex/4-captcha-solvers) |

Set `HF_TOKEN` in `.env` before `upload`.

## Citation

If you found this project useful, please cite it as:

```bibtex
@misc{zyukov2026_4captcha,
  author       = {Alex Zyukov},
  title        = {4-captcha: Synthetic Captcha Recognition and Adversarial Fine-tuning},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/pymlex/4-captcha}}
}
```

```bibtex
@article{dosovitskiy2020vit,
  title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author  = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal = {arXiv preprint arXiv:2010.11929},
  year    = {2020}
}
```

```bibtex
@article{goodfellow2014explaining,
  title   = {Explaining and Harnessing Adversarial Examples},
  author  = {Goodfellow, Ian J and Shlens, Jonathon and Szegedy, Christian},
  journal = {arXiv preprint arXiv:1412.6572},
  year    = {2014}
}
```

The project is under GPL-3.0 license.
