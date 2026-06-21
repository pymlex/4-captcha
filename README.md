# 4-captcha

Synthetic four-digit captcha recognition with Vision Transformer and compact CNN solvers, FGSM adversarial training, and exact-match evaluation.

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
| Adv train per model | 20,000 |
| Adv val per model | 1,000 |
| Adv test per model | 1,000 |

Grayscale synthetic four-digit captcha images for robust digit-string recognition and adversarial fine-tuning. Full release: [pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha) as `data.tar.gz` with `preview/` samples per split. Image format is $320 \times 80$ PNG with `labels.csv`. Per-digit TrueType fonts, elastic deformation, Bézier curves, Gaussian noise with $\sigma \sim \mathrm{Uniform}(5, 20)$, and model-specific FGSM splits under `adv/vit/` and `adv/cnn/`. Checkpoints: [pymlex/4-captcha-solvers](https://huggingface.co/pymlex/4-captcha-solvers).

## Project tree

```
4-captcha/
├── pyproject.toml
├── install.py
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
│   ├── publish.py
│   ├── upload_hf_dataset.py
│   ├── upload_hf_models.py
│   └── upload_hf.py
├── hub/
│   ├── dataset_card.md
│   ├── dataset_archive.py
│   ├── dataset_git_upload.py
│   ├── model_card.md
│   ├── github_publish.py
│   ├── hf_retry.py
│   └── publish.py
└── outputs/
    ├── metrics/
    ├── plots/
    └── predictions/
```

## Pipeline

```mermaid
flowchart TB
  subgraph generation [Dataset]
    G[Sample combos, render digits, global augment, PNG and labels]
  end
  subgraph train [Training]
    T[Clean train ViT and CNN, FGSM per model, fine-tune mixed 120k]
  end
  subgraph eval [Evaluation]
    E[Clean test 5k, adv test 1k, metrics and plots]
  end
  G --> T --> E
```

## Setup

One-command install after clone: pull updates, install dependencies, install githublex, authenticate GitHub and Hugging Face.

```bash
git clone https://github.com/pymlex/4-captcha.git
cd 4-captcha
python install.py
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

Render training curves, test comparison, and confusion matrices from outputs/.

```bash
python scripts/plot_results.py
```

## Publish

| Target | Repository |
|--------|------------|
| Metrics, predictions, plots, hub cards | [github.com/pymlex/4-captcha](https://github.com/pymlex/4-captcha) |
| Dataset archive, preview samples | [pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha) |
| Checkpoints, metrics, plots, model card | [pymlex/4-captcha-solvers](https://huggingface.co/pymlex/4-captcha-solvers) |

```bash
python scripts/publish.py
```

## Models

**CompactCaptchaNet** — four stride-2 conv blocks, reshape to $(1280, 20)$, `Conv1d` temporal mixing, adaptive pool to four positions, linear heads. About 1.4M parameters.

**CaptchaViT** — patch size $16 \times 16$, embed dim 256, depth 6, eight heads, learned position queries over patch tokens. About 4.8M parameters.

Clean training runs for 20 epochs. Adversarial fine-tuning runs for 20 epochs on a 120k mixed clean and adversarial set.

### Training curves

#### ViT clean

![ViT clean training loss](outputs/plots/vit_clean_loss.png)

The summed cross-entropy $L$ should fall steadily over 20 epochs. A persistent gap between train loss and `val_clean_loss` points to overfitting on font and noise patterns.

![ViT clean validation metrics](outputs/plots/vit_clean_val_metrics.png)

Validation exact match tracks the fraction of val images with a correct four-digit string.

#### ViT fine-tune

![ViT fine-tune training loss](outputs/plots/vit_finetune_loss.png)

Fine-tuning on the 120k mixed set should keep `val_clean_loss` near the clean checkpoint level while `val_adv_loss` drops relative to the clean-only model.

![ViT fine-tune validation metrics](outputs/plots/vit_finetune_val_metrics.png)

`val_adv_exact_match` measures robustness on FGSM images generated by the ViT clean checkpoint.

#### CNN clean

![CNN clean training loss](outputs/plots/cnn_clean_loss.png)

The CNN typically converges faster than the ViT because inductive locality matches fixed digit slots.

![CNN clean validation metrics](outputs/plots/cnn_clean_val_metrics.png)

CNN clean exact match on val is the reference for whether convolutional inductive bias helps on this rendering pipeline.

#### CNN fine-tune

![CNN fine-tune training loss](outputs/plots/cnn_finetune_loss.png)

The same mixed-set objective as ViT fine-tune.

![CNN fine-tune validation metrics](outputs/plots/cnn_finetune_val_metrics.png)

Compare `val_adv_exact_match` against the ViT fine-tune curve to see which architecture retains digit identity under FGSM.

### Test comparison

![Test model comparison](outputs/plots/test_model_comparison.png)

**Clean EM** — accuracy on 5,000 held-out clean test images.

**Adv EM** — accuracy on 1,000 FGSM test images for the matching model family.

**Robustness gap** — clean EM minus adv EM for the same checkpoint.

**Attack success rate** — fraction of clean-correct predictions that become wrong under FGSM.

### Confusion matrices

![ViT clean adv test](outputs/plots/confusion/vit_clean_adv_test_confusion.png)

![CNN clean adv test](outputs/plots/confusion/cnn_clean_adv_test_confusion.png)

Each heatmap aggregates predictions over all four digit positions into a single $10 \times 10$ count matrix for FGSM test images from the clean checkpoint.

Metrics JSON: `outputs/metrics/test_results.json`. Test predictions: `outputs/predictions/`.

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
