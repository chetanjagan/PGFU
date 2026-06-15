"""
utils.py
========
Shared utilities for PGFU training:
  - ModelEMA
  - FeatureHook
  - Cutout augmentation
  - Mixup
  - Learning rate scheduler
  - Gradient penalty
  - Loss functions
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Reproducibility ───────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── EMA ───────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    CRITICAL: copies BatchNorm running statistics (buffers) explicitly
    rather than exponentially averaging them. Without this fix, EMA
    model accuracy collapses to ~1% due to incorrect BN statistics.

    Args:
        model : student model to shadow
        decay : EMA decay factor (default 0.999)
    """
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # EMA update for parameters
        for ep, mp in zip(self.shadow.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        # Hard copy for BN running statistics (mean and variance buffers)
        for eb, mb in zip(self.shadow.buffers(), model.buffers()):
            eb.data.copy_(mb.data)

    def eval_model(self):
        return self.shadow


# ── Feature Hook ──────────────────────────────────────────────────────────
class FeatureHook:
    """
    Forward hook to capture intermediate layer outputs.

    Usage:
        hook = FeatureHook(model.layer4)
        output = model(x)
        features = hook.features  # shape: [B, C, H, W]
        hook.close()
    """
    def __init__(self, module):
        self.features = None
        self.hook     = module.register_forward_hook(self._fn)

    def _fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()
        self.features = None


# ── Cutout Augmentation ───────────────────────────────────────────────────
class Cutout:
    """
    Randomly zero out square patches in a tensor image.

    Args:
        n_holes : number of patches to cut out
        length  : side length of each patch
    """
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


# ── Mixup ─────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.4):
    """Apply Mixup augmentation to a batch."""
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(pred, ya, yb, lam, smoothing=0.1):
    """Cross-entropy loss for Mixup-augmented samples."""
    return (lam       * F.cross_entropy(pred, ya, label_smoothing=smoothing) +
            (1 - lam) * F.cross_entropy(pred, yb, label_smoothing=smoothing))


# ── LR Scheduler ──────────────────────────────────────────────────────────
def make_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Cosine annealing with linear warmup.

    Phase 1 (warmup): LR increases linearly from 0 to base LR.
    Phase 2 (cosine): LR decreases following cosine schedule to 0.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Loss Functions ────────────────────────────────────────────────────────
def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Knowledge Distillation loss (Hinton et al., 2015).

    L_KD = τ² · KL(σ(S/τ) || σ(T/τ))

    τ² scaling compensates for reduced gradient magnitude at high τ.
    """
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits    / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def uniform_confidence_loss(logits, num_classes):
    """
    Uniform Confidence loss: maximise entropy of forget-class predictions.

    L_UC = KL(σ(S(x_f)) || U_C) = log(C) - H(σ(S(x_f)))

    Minimising this drives forget-class outputs toward maximum uncertainty.
    """
    return F.kl_div(
        F.log_softmax(logits, dim=1),
        torch.full_like(logits, 1.0 / num_classes),
        reduction="batchmean",
    )


def gradient_penalty(critic, real_features, fake_features, device="cuda"):
    """
    WGAN-GP gradient penalty (Gulrajani et al., 2017).

    Enforces 1-Lipschitz constraint on critic by penalising
    gradient norm at interpolated points between real and fake.
    """
    batch_size = real_features.size(0)
    alpha      = torch.rand(batch_size, 1, device=device).expand_as(real_features)
    interpolated = (alpha * real_features + (1 - alpha) * fake_features
                    ).requires_grad_(True)
    output = critic(interpolated)
    grad   = torch.autograd.grad(
        output, interpolated,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# ── Quick Accuracy ────────────────────────────────────────────────────────
@torch.no_grad()
def quick_accuracy(model, loader, device="cuda", max_batches=50):
    """Fast approximate accuracy on first max_batches of loader."""
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y     = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0
