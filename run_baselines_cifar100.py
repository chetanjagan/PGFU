"""
baselines/run_baselines_cifar100.py
=====================================
All baseline unlearning methods for CIFAR-100.

Baselines implemented:
  1. Retrain     — gold standard oracle
  2. Gradient Ascent (GA)
  3. NegGrad
  4. Bad Teacher
  5. SCRUB       (Kurmanji et al., 2023)
  6. Fine-tune
  7. SISA        (Bourtoule et al., 2021)

All baselines use ResNet18 for fair comparison.

Usage:
    python baselines/run_baselines_cifar100.py \
        --teacher_path checkpoints/teacher_cifar100_resnet34.pth \
        --results_dir  results/cifar100_baselines \
        --forget_classes 0 1 2
"""

import argparse, copy, gc, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.resnet import cifar_resnet18, load_teacher
from utils import (ModelEMA, Cutout, make_cosine_scheduler,
                   kd_loss, quick_accuracy, set_seed)
from evaluate import compute_accuracy, compute_mia


NUM_CLASSES  = 100
MEAN         = (0.5071, 0.4867, 0.4408)
STD          = (0.2675, 0.2565, 0.2761)
LABEL_SMOOTH = 0.1
WARMUP       = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path",   type=str,   required=True)
    p.add_argument("--results_dir",    type=str,   default="results/cifar100_baselines")
    p.add_argument("--data_dir",       type=str,   default="./data")
    p.add_argument("--forget_classes", type=int,   nargs="+", default=[0, 1, 2])
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="cuda")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────
def get_loaders(args):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        Cutout(n_holes=1, length=8),
    ])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    trainset = torchvision.datasets.CIFAR100(args.data_dir, True,  download=True, transform=tf_train)
    testset  = torchvision.datasets.CIFAR100(args.data_dir, False, download=True, transform=tf_test)
    train_labels = torch.tensor(trainset.targets)
    test_labels  = torch.tensor(testset.targets)
    fs           = set(args.forget_classes)

    keep_idx        = [i for i,l in enumerate(train_labels) if l.item() not in fs]
    forget_idx      = [i for i,l in enumerate(train_labels) if l.item() in fs]
    test_keep_idx   = [i for i,l in enumerate(test_labels)  if l.item() not in fs]
    test_forget_idx = [i for i,l in enumerate(test_labels)  if l.item() in fs]
    full_idx        = list(range(len(trainset)))

    bs = args.batch_size
    return {
        "keep":        DataLoader(Subset(trainset, keep_idx),   bs, shuffle=True,  num_workers=0),
        "forget":      DataLoader(Subset(trainset, forget_idx), bs, shuffle=True,  num_workers=0),
        "full":        DataLoader(trainset,                     bs, shuffle=True,  num_workers=0),
        "test_keep":   DataLoader(Subset(testset, test_keep_idx),   256, shuffle=False, num_workers=0),
        "test_forget": DataLoader(Subset(testset, test_forget_idx), 256, shuffle=False, num_workers=0),
    }


def get_full_model(args, keep_loader, device, epochs=65):
    """Train / load the full model (starting point for all baselines)."""
    ckpt = os.path.join(args.results_dir, "full_model.pth")
    model = cifar_resnet18(NUM_CLASSES).to(device)
    if os.path.exists(ckpt):
        print("  Full model cached.", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model

    print("  Training full model...", flush=True)
    opt    = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, WARMUP, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model); best = 0.

    for epoch in range(1, epochs+1):
        model.train(); cor = tot = 0
        for x, y in tqdm(keep_loader, desc=f"  Full {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss   = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
            with torch.no_grad():
                cor += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
        sch.step()
        if epoch % 10 == 0 or epoch == epochs:
            acc = 100*cor/tot
            print(f"  Full {epoch}/{epochs}  acc={acc:.2f}%", flush=True)
            if acc > best: best = acc; torch.save(ema.shadow.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def evaluate_model(model, loaders, forget_classes, device):
    ka = compute_accuracy(model, loaders["test_keep"],   forget_classes, False, device)
    fa = compute_accuracy(model, loaders["test_forget"], forget_classes, True,  device)
    ml = compute_mia(model, loaders["forget"], loaders["test_forget"], "loss", device)
    mc = compute_mia(model, loaders["forget"], loaders["test_forget"], "conf", device)
    return {"keep": ka, "forget": fa, "mia_loss": ml, "mia_conf": mc}


# ── Baselines ─────────────────────────────────────────────────────────────

def run_retrain(args, loaders, device, epochs=65):
    name = "Retrain"
    path = os.path.join(args.results_dir, "retrain.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)

    model  = cifar_resnet18(NUM_CLASSES).to(device)
    opt    = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, WARMUP, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model); best = 0.
    ckpt   = os.path.join(args.results_dir, "retrain_model.pth")

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in tqdm(loaders["keep"], desc=f"  Retrain {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(model(x), y, label_smoothing=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        sch.step()
        acc = quick_accuracy(ema.eval_model(), loaders["test_keep"], device)
        if acc > best: best = acc; torch.save(ema.shadow.state_dict(), ckpt)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Retrain {epoch}/{epochs}  val={acc:.2f}%", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    r = {"name": name, **evaluate_model(model, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model; gc.collect(); torch.cuda.empty_cache()
    return r


def run_gradient_ascent(args, full_model, loaders, device, epochs=10, lr=1e-4):
    name = "Gradient Ascent"
    path = os.path.join(args.results_dir, "gradient_ascent.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)

    model  = copy.deepcopy(full_model)
    opt    = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model)

    for epoch in range(1, epochs+1):
        model.train()
        k_it = iter(loaders["keep"]); f_it = iter(loaders["forget"])
        steps = max(len(loaders["keep"]), len(loaders["forget"]))
        for _ in range(steps):
            try:    xf,yf = next(f_it)
            except: f_it = iter(loaders["forget"]); xf,yf = next(f_it)
            try:    xk,yk = next(k_it)
            except: k_it = iter(loaders["keep"]); xk,yk = next(k_it)
            xf,yf = xf.to(device),yf.to(device)
            xk,yk = xk.to(device),yk.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = -F.cross_entropy(model(xf),yf) + \
                        F.cross_entropy(model(xk),yk,label_smoothing=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        print(f"  GA {epoch}/{epochs} done", flush=True)

    r = {"name": name, **evaluate_model(ema.shadow, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model,ema; gc.collect(); torch.cuda.empty_cache()
    return r


def run_neggrad(args, full_model, loaders, device, epochs=10, lr=1e-4):
    name = "NegGrad"
    path = os.path.join(args.results_dir, "neggrad.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)
    # NegGrad = same as GA with retain regularisation
    model  = copy.deepcopy(full_model)
    opt    = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model)

    for epoch in range(1, epochs+1):
        model.train()
        k_it = iter(loaders["keep"]); f_it = iter(loaders["forget"])
        steps = max(len(loaders["keep"]), len(loaders["forget"]))
        for _ in range(steps):
            try:    xf,yf = next(f_it)
            except: f_it = iter(loaders["forget"]); xf,yf = next(f_it)
            try:    xk,yk = next(k_it)
            except: k_it = iter(loaders["keep"]); xk,yk = next(k_it)
            xf,yf = xf.to(device),yf.to(device)
            xk,yk = xk.to(device),yk.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = -F.cross_entropy(model(xf),yf) + \
                        F.cross_entropy(model(xk),yk,label_smoothing=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        print(f"  NegGrad {epoch}/{epochs} done", flush=True)

    r = {"name": name, **evaluate_model(ema.shadow, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model,ema; gc.collect(); torch.cuda.empty_cache()
    return r


def run_bad_teacher(args, full_model, teacher, loaders, device, epochs=10):
    name = "Bad Teacher"
    path = os.path.join(args.results_dir, "bad_teacher.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)

    bad_teacher = cifar_resnet18(NUM_CLASSES).to(device).eval()
    for p in bad_teacher.parameters(): p.requires_grad_(False)

    model  = copy.deepcopy(full_model)
    opt    = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model)
    TEMP   = 4.; GAMMA = 0.9; ALPHA = 0.1

    def kl(s,t,temp=TEMP):
        return F.kl_div(F.log_softmax(s/temp,dim=1),
                        F.softmax(t/temp,dim=1), reduction="batchmean") * temp**2

    for epoch in range(1, epochs+1):
        model.train()
        k_it = iter(loaders["keep"]); f_it = iter(loaders["forget"])
        steps = max(len(loaders["keep"]), len(loaders["forget"]))
        for _ in range(steps):
            try:    xk,yk = next(k_it)
            except: k_it = iter(loaders["keep"]); xk,yk = next(k_it)
            try:    xf,yf = next(f_it)
            except: f_it = iter(loaders["forget"]); xf,yf = next(f_it)
            xk,yk = xk.to(device),yk.to(device)
            xf,yf = xf.to(device),yf.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    good_k = teacher(xk); bad_f = bad_teacher(xf)
                slk = model(xk); slf = model(xf)
                loss = GAMMA*kl(slk,good_k) + ALPHA*kl(slf,bad_f)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        print(f"  Bad Teacher {epoch}/{epochs} done", flush=True)

    r = {"name": name, **evaluate_model(ema.shadow, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model,ema,bad_teacher; gc.collect(); torch.cuda.empty_cache()
    return r


def run_scrub(args, full_model, teacher, loaders, device, epochs=10):
    """
    SCRUB (Kurmanji et al., NeurIPS 2023).
    Hyperparameters from original paper:
      lr=0.0001, alpha=0.001, gamma=0.99, temp=4.0
    """
    name = "SCRUB"
    path = os.path.join(args.results_dir, "scrub.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)

    model  = copy.deepcopy(full_model)
    # Original paper: lr=0.0001
    opt    = torch.optim.SGD(model.parameters(), lr=0.0001,
                             momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, 2, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model)
    TEMP   = 4.; GAMMA = 0.99; ALPHA = 0.001

    def kl(s, t, temp=TEMP):
        return F.kl_div(F.log_softmax(s/temp,dim=1),
                        F.softmax(t/temp,dim=1), reduction="batchmean") * temp**2

    for epoch in range(1, epochs+1):
        model.train()
        k_it = iter(loaders["keep"]); f_it = iter(loaders["forget"])
        steps = max(len(loaders["keep"]), len(loaders["forget"]))
        for _ in range(steps):
            try:    xk,yk = next(k_it)
            except: k_it = iter(loaders["keep"]); xk,yk = next(k_it)
            try:    xf,yf = next(f_it)
            except: f_it = iter(loaders["forget"]); xf,yf = next(f_it)
            xk,yk = xk.to(device),yk.to(device)
            xf,yf = xf.to(device),yf.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad(): tlk = teacher(xk); tlf = teacher(xf)
                slk = model(xk); slf = model(xf)
                loss = (GAMMA*kl(slk,tlk) +
                        ALPHA*(-kl(slf,tlf)) +
                        F.cross_entropy(slk,yk,label_smoothing=LABEL_SMOOTH))
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        sch.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  SCRUB {epoch}/{epochs} done", flush=True)

    r = {"name": name, **evaluate_model(ema.shadow, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model,ema; gc.collect(); torch.cuda.empty_cache()
    return r


def run_finetune(args, full_model, loaders, device, epochs=20):
    name = "Fine-tune"
    path = os.path.join(args.results_dir, "finetune.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)

    model  = copy.deepcopy(full_model)
    opt    = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, 2, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(model)

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in tqdm(loaders["keep"],
                         desc=f"  Finetune {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(model(x), y, label_smoothing=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        sch.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Finetune {epoch}/{epochs} done", flush=True)

    r = {"name": name, **evaluate_model(ema.shadow, loaders, args.forget_classes, device)}
    with open(path, "w") as f: json.dump(r, f, indent=2)
    print_result(r); del model,ema; gc.collect(); torch.cuda.empty_cache()
    return r


def print_result(r):
    print(f"  {r['name']:<20}  "
          f"Keep={r['keep']:.2f}%  "
          f"Forget={r['forget']:.2f}%  "
          f"MIA-loss={r['mia_loss']:.4f}", flush=True)


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Baselines — CIFAR-100 | Forget: {args.forget_classes}")
    loaders = get_loaders(args)
    teacher = load_teacher(args.teacher_path, "resnet34", NUM_CLASSES, device)
    full    = get_full_model(args, loaders["keep"], device)

    results = []
    results.append(run_retrain(args, loaders, device))
    results.append(run_gradient_ascent(args, full, loaders, device))
    results.append(run_neggrad(args, full, loaders, device))
    results.append(run_bad_teacher(args, full, teacher, loaders, device))
    results.append(run_scrub(args, full, teacher, loaders, device))
    results.append(run_finetune(args, full, loaders, device))

    # Final summary
    print(f"\n{'='*65}")
    print(f"  CIFAR-100 BASELINES (forget classes: {args.forget_classes})")
    print(f"{'='*65}")
    print(f"  {'Method':<22} {'Keep':>8} {'Forget':>8} {'MIA-loss':>10}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['name']:<22} {r['keep']:>7.2f}%  "
              f"{r['forget']:>7.2f}%  {r['mia_loss']:>10.4f}")

    with open(os.path.join(args.results_dir, "all_baselines.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
