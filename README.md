# PGFU: Prototype-Guided Feature Unlearning

> **AAAI 2026 Submission**
> Anonymous Authors

---

## Overview

**PGFU** (Prototype-Guided Feature Unlearning) is a two-stage machine unlearning framework that achieves exact class-level unlearning through feature-level manipulation — not merely output-level suppression.

### Key Results

| Dataset | Keep Acc ↑ | Forget Acc ↓ | MIA-loss |
|---|---|---|---|
| CIFAR-10 | 95.15% | **0.00%** | 0.5407 |
| CIFAR-100 | 76.78% | **0.00%** | 0.5185 |
| TinyImageNet-200 | 62.53% | **0.00%** | 0.6203 |

PGFU is the **only method** achieving FA=0.00% while maintaining competitive retain accuracy across all three benchmarks against 7 baselines.

---

## Method

PGFU operates in two stages:

**Stage 1 — Retain-Set Pretraining:**
The ResNet18 student is trained on the retain set $\mathcal{D}_r$ using cross-entropy with label smoothing and Mixup augmentation, building a solid retain-class feature geometry before unlearning begins.

**Stage 2 — Unlearning via Push-Pull:**
Five complementary losses are jointly optimised:
- **KD Loss** — knowledge distillation from frozen ResNet34 teacher on retain samples
- **CE Loss** — Mixup cross-entropy on retain samples  
- **UC Loss** — entropy maximisation on forget-class outputs
- **Pull Loss** — cosine similarity maximisation between retain features and teacher prototypes
- **Push Loss** — cosine repulsion of forget features away from teacher prototypes (ReLU-gated, confidence-weighted, linearly ramped)

An adversarial WGAN-GP critic additionally aligns forget-class feature distributions for MIA indistinguishability.

### Architecture

```
Frozen Teacher (ResNet34) ──→ Prototypes + KD targets + Confidence weights
                                      ↓
                          Student (ResNet18) ← Stage 2: Push + Pull + KD + CE + UC + Critic
                                      ↓
                              EMA Model (final)
```

---

## Installation

```bash
git clone https://github.com/anonymous/PGFU.git
cd PGFU
pip install -r requirements.txt
```

---

## Dataset Setup

**CIFAR-10 / CIFAR-100** — downloaded automatically via torchvision.

**TinyImageNet-200:**
```bash
mkdir -p data/tiny-imagenet-200
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

---

## Training

### CIFAR-100 (3 forget classes)

```bash
python pgfu_cifar100.py \
    --data_dir ./data \
    --teacher_path ./checkpoints/teacher_cifar100_resnet34.pth \
    --results_dir ./results/cifar100 \
    --forget_classes 0 1 2 \
    --stage1_epochs 25 \
    --stage2_epochs 40 \
    --seed 42
```

### CIFAR-10 (2 forget classes)

```bash
python pgfu_cifar10.py \
    --data_dir ./data \
    --teacher_path ./checkpoints/teacher_cifar10.pth \
    --results_dir ./results/cifar10 \
    --forget_classes 0 1 \
    --stage1_epochs 25 \
    --stage2_epochs 40 \
    --seed 42
```

### TinyImageNet-200 (10 forget classes)

```bash
python pgfu_tinyimagenet.py \
    --data_dir ./data/tiny-imagenet-200 \
    --teacher_path ./checkpoints/best.pth \
    --results_dir ./results/tinyimagenet \
    --forget_classes 0 1 2 3 4 5 6 7 8 9 \
    --stage1_epochs 30 \
    --stage2_epochs 40 \
    --seed 42
```

---

## Evaluation

```bash
# Standard evaluation (keep acc, forget acc, MIA)
python evaluate.py \
    --checkpoint ./results/cifar100/pgfu_ema_final.pth \
    --dataset cifar100 \
    --forget_classes 0 1 2

# Linear probe (feature-level unlearning verification)
python linear_probe.py \
    --checkpoint ./results/cifar100/pgfu_ema_final.pth \
    --dataset cifar100 \
    --forget_classes 0 1 2
```

---

## Hyperparameters

### CIFAR-10 / CIFAR-100

| Parameter | Value |
|---|---|
| Stage 1 epochs | 25 |
| Stage 2 epochs | 40 |
| Warmup epochs | 5 |
| LR Stage 1 | 0.05 |
| LR Stage 2 | 0.01 |
| KD temperature τ | 4.0 |
| Label smoothing ε | 0.1 |
| λ_KD | 1.0 |
| λ_CE | 1.0 |
| λ_UC | 0.005 |
| λ_pull | 0.01 |
| λ_push (max) | 0.15 |
| λ_ADV | 0.0002 |
| λ_GP | 5.0 |
| EMA decay δ | 0.999 |

### TinyImageNet-200

Same as above except:
- LR Stage 2 = 0.001
- λ_push (max) = 0.40
- λ_ADV = 0.0 (critic disabled for stability)

---

## Checkpoints

Pre-trained teacher checkpoints:

| File | Dataset | Architecture | Accuracy |
|---|---|---|---|
| `teacher_cifar10.pth` | CIFAR-10 | ResNet34 | ~95.6% |
| `teacher_cifar100_resnet34.pth` | CIFAR-100 | ResNet34 | ~77.5% |
| `best.pth` | TinyImageNet | ResNet34 | ~64.2% |

---

## Baselines

All baselines (GA, NegGrad, Bad Teacher, SCRUB, Fine-tune, SISA) are implemented in `baselines/` and use the identical ResNet18 architecture for fair comparison.

```bash
python baselines/run_all_baselines.py \
    --dataset cifar100 \
    --forget_classes 0 1 2
```

---

## Results Reproduction

To reproduce all paper results:

```bash
bash scripts/reproduce_all.sh
```

This runs PGFU and all baselines on all three datasets with seeds 42, 123, 456.

---

## Citation

```bibtex
@inproceedings{pgfu2026,
  title     = {PGFU: Prototype-Guided Feature Unlearning for Deep Neural Networks},
  author    = {Anonymous},
  booktitle = {AAAI 2026},
  year      = {2026}
}
```

---

## License

MIT License
