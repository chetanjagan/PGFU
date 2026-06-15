"""
evaluate.py
===========
Standard evaluation for any PGFU or baseline checkpoint.

Computes:
  - Keep Accuracy  (test samples from retain classes)
  - Forget Accuracy (test samples from forget classes)
  - MIA-loss score (logistic regression on per-sample loss)
  - MIA-conf score (logistic regression on max confidence)

Usage:
    python evaluate.py \
        --checkpoint results/cifar100/pgfu_ema_final.pth \
        --dataset cifar100 \
        --forget_classes 0 1 2

    python evaluate.py \
        --checkpoint results/tinyimagenet/pgfu_ema_final.pth \
        --dataset tinyimagenet \
        --forget_classes 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from models.resnet import cifar_resnet18, tiny_resnet18
from datasets.tinyimagenet import TinyImageNetDataset, get_tinyimagenet_transforms


# ── Arguments ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PGFU Evaluation")
    p.add_argument("--checkpoint",     type=str,   required=True)
    p.add_argument("--dataset",        type=str,   required=True,
                   choices=["cifar10", "cifar100", "tinyimagenet"])
    p.add_argument("--data_dir",       type=str,   default="./data")
    p.add_argument("--forget_classes", type=int,   nargs="+", required=True)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--output",         type=str,   default=None)
    return p.parse_args()


# ── Data loaders ──────────────────────────────────────────────────────────
def get_loaders(args):
    forget_set = set(args.forget_classes)

    if args.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.CIFAR10(args.data_dir, True,  download=True, transform=tf)
        testset  = torchvision.datasets.CIFAR10(args.data_dir, False, download=True, transform=tf)
        nc       = 10

    elif args.dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.CIFAR100(args.data_dir, True,  download=True, transform=tf)
        testset  = torchvision.datasets.CIFAR100(args.data_dir, False, download=True, transform=tf)
        nc       = 100

    else:  # tinyimagenet
        _, tf = get_tinyimagenet_transforms()
        trainset = TinyImageNetDataset(args.data_dir, "train", tf)
        testset  = TinyImageNetDataset(args.data_dir, "val",   tf)
        nc       = 200

    # Get labels
    if hasattr(trainset, "targets"):
        train_labels = torch.tensor(trainset.targets)
        test_labels  = torch.tensor(testset.targets)
    else:
        train_labels = torch.tensor([s[1] for s in trainset.samples])
        test_labels  = torch.tensor([s[1] for s in testset.samples])

    # Split indices
    forget_train_idx = [i for i, l in enumerate(train_labels) if l.item() in forget_set]
    test_keep_idx    = [i for i, l in enumerate(test_labels)  if l.item() not in forget_set]
    test_forget_idx  = [i for i, l in enumerate(test_labels)  if l.item() in forget_set]

    bs = args.batch_size
    forget_loader      = DataLoader(Subset(trainset, forget_train_idx),
                                    bs, shuffle=False, num_workers=0)
    test_keep_loader   = DataLoader(Subset(testset, test_keep_idx),
                                    bs, shuffle=False, num_workers=0)
    test_forget_loader = DataLoader(Subset(testset, test_forget_idx),
                                    bs, shuffle=False, num_workers=0)

    return forget_loader, test_keep_loader, test_forget_loader, nc


# ── Accuracy ───────────────────────────────────────────────────────────────
def compute_accuracy(model, loader, forget_classes, mask_forget, device):
    fc    = torch.tensor(forget_classes, device=device)
    model.eval()
    cor = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            m     = torch.isin(y, fc) if mask_forget else ~torch.isin(y, fc)
            if m.sum() == 0: continue
            cor += (preds[m] == y[m]).sum().item()
            tot += m.sum().item()
    return 100.0 * cor / tot if tot > 0 else 0.0


# ── MIA ───────────────────────────────────────────────────────────────────
def compute_mia(model, member_loader, nonmember_loader, mode, device):
    """
    Membership Inference Attack via logistic regression.
    mode: 'loss' or 'conf'
    Returns AUC score (0.5 = ideal indistinguishability).
    """
    model.eval()
    member_vals = []; nonmember_vals = []

    with torch.no_grad():
        for x, y in member_loader:
            x, y = x.to(device), y.to(device)
            if mode == "loss":
                member_vals.extend(
                    F.cross_entropy(model(x), y, reduction="none").cpu().numpy())
            else:
                member_vals.extend(
                    F.softmax(model(x), dim=1).max(1)[0].cpu().numpy())
            if len(member_vals) >= 2000: break

        for x, y in nonmember_loader:
            x, y = x.to(device), y.to(device)
            if mode == "loss":
                nonmember_vals.extend(
                    F.cross_entropy(model(x), y, reduction="none").cpu().numpy())
            else:
                nonmember_vals.extend(
                    F.softmax(model(x), dim=1).max(1)[0].cpu().numpy())
            if len(nonmember_vals) >= 2000: break

    n = min(len(member_vals), len(nonmember_vals))
    if n < 50: return 0.5

    X = np.vstack([np.array(member_vals[:n]).reshape(-1, 1),
                   np.array(nonmember_vals[:n]).reshape(-1, 1)])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                            stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Dataset    : {args.dataset}", flush=True)
    print(f"Checkpoint : {args.checkpoint}", flush=True)
    print(f"Forget     : {args.forget_classes}", flush=True)

    # Load data
    forget_loader, test_keep_loader, test_forget_loader, nc = get_loaders(args)

    # Load model
    if args.dataset == "tinyimagenet":
        model = tiny_resnet18(nc)
    else:
        model = cifar_resnet18(nc)

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Evaluate
    keep_acc   = compute_accuracy(model, test_keep_loader,
                                   args.forget_classes, False, device)
    forget_acc = compute_accuracy(model, test_forget_loader,
                                   args.forget_classes, True, device)
    mia_loss   = compute_mia(model, forget_loader, test_forget_loader,
                              "loss", device)
    mia_conf   = compute_mia(model, forget_loader, test_forget_loader,
                              "conf", device)

    results = {
        "checkpoint":   args.checkpoint,
        "dataset":      args.dataset,
        "forget_classes": args.forget_classes,
        "keep_acc":     round(keep_acc,   4),
        "forget_acc":   round(forget_acc, 4),
        "mia_loss":     round(mia_loss,   4),
        "mia_conf":     round(mia_conf,   4),
    }

    print(f"\n{'='*50}")
    print(f"  Keep Accuracy  : {keep_acc:.2f}%")
    print(f"  Forget Accuracy: {forget_acc:.2f}%")
    print(f"  MIA-loss       : {mia_loss:.4f}  (ideal=0.5)")
    print(f"  MIA-conf       : {mia_conf:.4f}  (ideal=0.5)")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
