"""
models/resnet.py
================
Modified ResNet architectures for CIFAR (32x32) and
TinyImageNet (64x64) inputs.

Key modifications from standard ResNet:
  - 7x7 conv stem replaced with 3x3 (preserves spatial resolution)
  - MaxPool replaced with Identity (no early downsampling)
"""

import torch
import torch.nn as nn
from torchvision import models


def cifar_resnet18(num_classes=100):
    """ResNet18 adapted for 32x32 CIFAR inputs."""
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


def cifar_resnet34(num_classes=100):
    """ResNet34 adapted for 32x32 CIFAR inputs (used as teacher)."""
    model = models.resnet34(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


def tiny_resnet18(num_classes=200):
    """ResNet18 adapted for 64x64 TinyImageNet inputs."""
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


def tiny_resnet34(num_classes=200):
    """ResNet34 adapted for 64x64 TinyImageNet inputs (used as teacher)."""
    model = models.resnet34(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


def load_teacher(path, architecture="resnet34", num_classes=100, device="cuda"):
    """
    Load a pretrained teacher checkpoint.

    Handles three checkpoint formats:
      1. Direct state_dict  (OrderedDict)
      2. {'model_state_dict': ...}
      3. {'state_dict': ...}

    Args:
        path         : path to .pth file
        architecture : 'resnet34' or 'resnet18'
        num_classes  : 100 (CIFAR-100) or 200 (TinyImageNet)
        device       : 'cuda' or 'cpu'

    Returns:
        Frozen teacher model on device.
    """
    if num_classes == 100:
        model = (cifar_resnet34(num_classes) if architecture == "resnet34"
                 else cifar_resnet18(num_classes))
    else:
        model = (tiny_resnet34(num_classes) if architecture == "resnet34"
                 else tiny_resnet18(num_classes))

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


class Projection(nn.Module):
    """
    1x1 conv projection network mapping student layer4 channels
    to teacher layer4 channels for prototype-based losses.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class StrongCritic(nn.Module):
    """
    4-layer MLP Wasserstein critic for adversarial feature alignment.
    Input: flattened projected layer4 features.
    Output: scalar score.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)
