# 4-captcha

Synthetic four-digit captcha recognition with Vision Transformer and compact CNN solvers, FGSM adversarial training, and per-position evaluation.

## Overview

The pipeline generates grayscale $320 \times 80$ captcha images with per-digit font variation, elastic deformation, Bézier noise curves, and FGSM perturbations. Two classifiers output logits $\mathbf{Z} \in \mathbb{R}^{B \times 4 \times 10}$. The training objective is

$$\mathcal{L} = \sum_{p=1}^{4} \mathrm{CE}(\mathbf{Z}_{:,p,:}, y_p)$$

FGSM examples follow

$$x_{\mathrm{adv}} = \mathrm{clip}\bigl(x + \varepsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}),\, 0,\, 1\bigr)$$

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
    G1[Sample combos] --> G2[Render digits]
    G2 --> G3[Global augment]
    G3 --> G4[PNG + labels]
  end
  subgraph train [Training]
    T1[Clean train ViT/CNN] --> T2[FGSM per model]
    T2 --> T3[Fine-tune mixed 120k]
  end
  subgraph eval [Evaluation]
    E1[Clean test 5k] --> E3[Metrics + plots]
    E2[Adv test 1k] --> E3
  end
  G4 --> T1
  T3 --> E1
  T3 --> E2
```

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
```

Defaults target the full run: 132,000 images, `CLEAN_EPOCHS=50`, `FINETUNE_EPOCHS=1`, `QUICK_MODE=false`. Set `QUICK_MODE=true` only for local pipeline smoke tests.

## Scripts

| Script | Description |
|--------|-------------|
| `main.py` | Pipeline entry point. `--step` selects a stage, `--model` selects `vit`, `cnn`, or `both`. |
| `scripts/generate_dataset.py` | Generate clean train, val, and test PNGs and `labels.csv` under `data/clean/`. |
| `scripts/train_clean.py` | Train ViT or CNN on clean data. `--model {vit,cnn}`. |
| `scripts/generate_fgsm.py` | Build FGSM splits from a trained clean model. `--model {vit,cnn}`. |
| `scripts/finetune.py` | Fine-tune on clean plus adversarial train mix. `--model {vit,cnn}`. |
| `scripts/evaluate.py` | Run test metrics and save predictions. |
| `scripts/plot_results.py` | Render evaluation and training plots from `outputs/`. |
| `scripts/upload_hf.py` | Push dataset and checkpoints to Hugging Face. Requires `HF_TOKEN`. |

```bash
python main.py --step all
python main.py --step train_clean --model vit
python scripts/generate_dataset.py
python scripts/train_clean.py --model cnn
python scripts/generate_fgsm.py --model vit
python scripts/finetune.py --model vit
python scripts/evaluate.py
python scripts/plot_results.py
python scripts/upload_hf.py
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
