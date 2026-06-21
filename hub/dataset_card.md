---
license: gpl-3.0
language:
- en
tags:
- captcha
- ocr
- synthetic
- adversarial
size_categories:
- 100K<n<1M
---

# 4-captcha dataset

Grayscale synthetic four-digit captcha images for robust digit-string recognition and adversarial fine-tuning experiments.

## Splits

| Split | Images | Labels |
|-------|--------|--------|
| clean/train | 100,000 | 4-digit string |
| clean/val | 5,000 | 4-digit string |
| clean/test | 5,000 | 4-digit string |
| adv/{vit,cnn}/train | 20,000 each | same as source clean image |
| adv/{vit,cnn}/val | 1,000 each | same as source clean image |
| adv/{vit,cnn}/test | 1,000 each | same as source clean image |

Total stored images: 132,000 for a full release with both ViT and CNN adversarial splits.

Digit combinations for clean train are sampled with replacement from an 8,000-combo pool. Clean val and test use disjoint 1,000-combo pools. Adversarial splits are generated per model from trained clean checkpoints with FGSM.

## Image format

- Resolution: $320 \times 80$ pixels
- Channels: 1, grayscale PNG
- Label file: `labels.csv` with columns `filename`, `label`

## Rendering

Each image contains four digits $d_1 d_2 d_3 d_4$, $d_i \in \{0,\ldots,9\}$.

Per-digit variation:
- independent TrueType font from a system font pool
- native glyph height about $0.45$–$0.55$ of image height
- rotation uniform in $[-15°, 15°]$
- small horizontal and vertical shift
- optional slight overlap between neighbours

Canvas background is white or light gray.

## Global transforms

Applied to the full image after digit placement, with seed 42:
- elastic deformation
- 2–4 dark Bézier curves, thickness 1–3 px
- Gaussian noise with $\sigma \sim \mathrm{Uniform}(5, 20)$ on the 0–255 scale
- Gaussian blur with kernel $3 \times 3$ or $5 \times 5$
- brightness and contrast jitter
- gamma scaling

## Adversarial splits

FGSM with

$$x_{adv} = clip(x + \varepsilon \cdot sign(\nabla_x L), 0, 1)$$

where $L$ is the sum of four cross-entropy terms over digit positions.

Training adversarial images use $\varepsilon \in \{0.015, 0.03\}$ in equal proportion. Validation and test adversarial images use the same $\varepsilon$ set per split. Each model has its own adversarial folders under `adv/vit/` and `adv/cnn/`.

## Layout

```
data/
├── clean/
│   ├── train/
│   ├── val/
│   └── test/
└── adv/
    ├── vit/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── cnn/
        ├── train/
        ├── val/
        └── test/
```

## Code and training

Pipeline source: [github.com/pymlex/4-captcha](https://github.com/pymlex/4-captcha)

Model checkpoints and evaluation artefacts: [pymlex/4-captcha-solvers](https://huggingface.co/pymlex/4-captcha-solvers)

## Citation

```bibtex
@misc{zyukov2026_4captcha,
  author       = {Alex Zyukov},
  title        = {4-captcha: Synthetic Captcha Recognition and Adversarial Fine-tuning},
  year         = {2026},
  howpublished = {\url{https://github.com/pymlex/4-captcha}}
}
```

The dataset is under GPL-3.0 license.
