"""
linear_probe.py
===============
Correct linear probe evaluation for feature-level unlearning verification.

Protocol (CORRECT):
  1. Train linear classifier on RETAIN TRAIN features ONLY
  2. Test on retain TEST features  → keep probe accuracy
  3. Test on forget TEST features  → forget probe accuracy

If forget probe accuracy ≈ 0.5% (1/C random chance):
  → Genuine feature-level unlearning achieved

If forget probe accuracy is high despite zero forget accuracy:
  → Output-level forgetting only; internal features persist

Usage:
    python linear_probe.py \
        --checkpoint results/cifar100/pgfu_ema_final.pth \
        --dataset cifar100 \
        --forget_classes 0 1 2

    # Compare PGFU vs Retrain side by side
    python linear_probe.py \
        --checkpoint results/cifar100/pgfu_ema_final.pth \
        --retrain_checkpoint results/cifar100/retrain_best_ema.pth \
        --dataset cifar100 \
        --forget_classes 0 1 2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from models.resnet import cifar_resnet18, tiny_resnet18
from datasets.tinyimagenet import TinyImageNetDataset, get_tinyimagenet_transforms


def parse_args():
    p = argparse.ArgumentParser(description="Linear Probe Evaluation")
    p.add_argument("--checkpoint",          type=str, required=True)
    p.add_argument("--retrain_checkpoint",  type=str, default=None,
                   help="Optional retrain checkpoint for comparison")
    p.add_argument("--dataset",             type=str, required=True,
                   choices=["cifar10", "cifar100", "tinyimagenet"])
    p.add_argument("--data_dir",            type=str, default="./data")
    p.add_argument("--forget_classes",      type=int, nargs="+", required=True)
    p.add_argument("--batch_size",          type=int, default=256)
    p.add_argument("--max_train_samples",   type=int, default=50000)
    p.add_argument("--device",              type=str, default="cuda")
    p.add_argument("--output",              type=str, default=None)
    return p.parse_args()


def get_data(args):
    forget_set = set(args.forget_classes)

    # No augmentation for feature extraction
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
    else:
        _, tf    = get_tinyimagenet_transforms()
        trainset = TinyImageNetDataset(args.data_dir, "train", tf)
        testset  = TinyImageNetDataset(args.data_dir, "val",   tf)
        nc       = 200

    if hasattr(trainset, "targets"):
        train_labels = torch.tensor(trainset.targets)
        test_labels  = torch.tensor(testset.targets)
    else:
        train_labels = torch.tensor([s[1] for s in trainset.samples])
        test_labels  = torch.tensor([s[1] for s in testset.samples])

    train_keep_idx  = [i for i, l in enumerate(train_labels) if l.item() not in forget_set]
    test_keep_idx   = [i for i, l in enumerate(test_labels)  if l.item() not in forget_set]
    test_forget_idx = [i for i, l in enumerate(test_labels)  if l.item() in forget_set]

    bs = args.batch_size
    train_keep_loader  = DataLoader(Subset(trainset, train_keep_idx),
                                    bs, shuffle=False, num_workers=0)
    test_keep_loader   = DataLoader(Subset(testset, test_keep_idx),
                                    bs, shuffle=False, num_workers=0)
    test_forget_loader = DataLoader(Subset(testset, test_forget_idx),
                                    bs, shuffle=False, num_workers=0)

    return train_keep_loader, test_keep_loader, test_forget_loader, nc


def extract_features(model, loader, device, max_samples=50000):
    """Extract avgpool (512-dim) features before the FC layer."""
    model.eval()
    feats  = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Use avgpool output (after layer4, before FC)
            feat = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(
                                model.relu(
                                    model.bn1(
                                        model.conv1(x))))))))
            feats.append(feat.flatten(1).cpu().numpy())
            labels.extend(y.numpy())
            if len(labels) >= max_samples:
                break

    return (np.vstack(feats)[:max_samples],
            np.array(labels[:max_samples]))


def run_probe(model, name, train_keep_loader, test_keep_loader,
              test_forget_loader, device, max_train_samples):
    """Run the correct linear probe protocol for one model."""
    print(f"\n  [{name}] Extracting features...", flush=True)

    X_train,  y_train  = extract_features(model, train_keep_loader,
                                           device, max_train_samples)
    X_keep,   y_keep   = extract_features(model, test_keep_loader, device)
    X_forget, y_forget = extract_features(model, test_forget_loader, device)

    # Standardise using retain train statistics
    scaler    = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_keep_s   = scaler.transform(X_keep)
    X_forget_s = scaler.transform(X_forget)

    print(f"  [{name}] Training linear probe on retain features...", flush=True)
    clf = LogisticRegression(max_iter=2000, C=0.1,
                              random_state=42, solver="lbfgs",
                              multi_class="multinomial", n_jobs=-1)
    clf.fit(X_train_s, y_train)

    keep_acc   = 100.0 * clf.score(X_keep_s,   y_keep)
    forget_acc = 100.0 * clf.score(X_forget_s, y_forget)

    print(f"  [{name}] Keep probe   : {keep_acc:.2f}%", flush=True)
    print(f"  [{name}] Forget probe : {forget_acc:.2f}%", flush=True)

    return {"keep_probe": keep_acc, "forget_probe": forget_acc}


def load_model(checkpoint, nc, dataset, device):
    if dataset == "tinyimagenet":
        model = tiny_resnet18(nc)
    else:
        model = cifar_resnet18(nc)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Dataset : {args.dataset}")
    print(f"Forget  : {args.forget_classes}")

    train_keep_loader, test_keep_loader, test_forget_loader, nc = get_data(args)

    results = {}

    # Main checkpoint
    model = load_model(args.checkpoint, nc, args.dataset, device)
    results["pgfu"] = run_probe(model, "PGFU",
                                train_keep_loader, test_keep_loader,
                                test_forget_loader, device,
                                args.max_train_samples)
    del model

    # Optional retrain comparison
    if args.retrain_checkpoint:
        model = load_model(args.retrain_checkpoint, nc, args.dataset, device)
        results["retrain"] = run_probe(model, "Retrain",
                                       train_keep_loader, test_keep_loader,
                                       test_forget_loader, device,
                                       args.max_train_samples)
        del model

    # Summary
    print(f"\n{'='*55}")
    print("  CORRECT LINEAR PROBE RESULTS")
    print(f"{'='*55}")
    print(f"  {'Model':<15} {'Keep Probe':>12} {'Forget Probe':>14}")
    print(f"  {'-'*45}")
    for name, r in results.items():
        print(f"  {name:<15} {r['keep_probe']:>11.2f}%  "
              f"{r['forget_probe']:>13.2f}%")
    print(f"\n  Note: Random chance = {100/nc:.1f}%  (1/{nc} classes)")
    print(f"  If PGFU forget probe ≈ random chance → feature-level unlearning ✓")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
