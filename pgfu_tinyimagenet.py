"""
pgfu_tinyimagenet.py
====================
PGFU Unlearning — TinyImageNet-200

Teacher : ResNet34 loaded from best.pth
           (Format: {'model_state_dict': ...})
Student : ResNet18

Key differences from CIFAR version:
  - LAMBDA_ADV = 0.0  (critic disabled for 64x64 stability)
  - LAMBDA_PUSH = 0.40 (stronger push needed for larger dataset)
  - LR_STAGE2  = 0.001 (critical — 0.01 causes collapse)
  - Stage 1: 30 epochs; Stage 2: 40 epochs

Usage:
    python pgfu_tinyimagenet.py \
        --teacher_path checkpoints/best.pth \
        --data_dir     data/tiny-imagenet-200 \
        --results_dir  results/tinyimagenet \
        --forget_classes 0 1 2 3 4 5 6 7 8 9 \
        --seed 42
"""

import argparse, gc, json, os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models.resnet import tiny_resnet18, tiny_resnet34, Projection, load_teacher
from datasets.tinyimagenet import TinyImageNetDataset, get_tinyimagenet_transforms
from utils import (ModelEMA, FeatureHook, Cutout, mixup_batch, mixup_criterion,
                   make_cosine_scheduler, kd_loss, uniform_confidence_loss,
                   quick_accuracy, set_seed)
from evaluate import compute_accuracy, compute_mia


NUM_CLASSES = 200


def parse_args():
    p = argparse.ArgumentParser(description="PGFU TinyImageNet-200")
    p.add_argument("--teacher_path",   type=str,   required=True)
    p.add_argument("--data_dir",       type=str,   required=True)
    p.add_argument("--results_dir",    type=str,   default="results/tinyimagenet")
    p.add_argument("--forget_classes", type=int,   nargs="+",
                   default=list(range(10)))
    p.add_argument("--stage1_epochs",  type=int,   default=30)
    p.add_argument("--stage2_epochs",  type=int,   default=40)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr_stage1",      type=float, default=0.05)
    p.add_argument("--lr_stage2",      type=float, default=0.001)
    p.add_argument("--lambda_kd",      type=float, default=1.0)
    p.add_argument("--lambda_ce",      type=float, default=1.0)
    p.add_argument("--lambda_uc",      type=float, default=0.005)
    p.add_argument("--lambda_pull",    type=float, default=0.01)
    p.add_argument("--lambda_push",    type=float, default=0.40)
    p.add_argument("--lambda_adv",     type=float, default=0.0)
    p.add_argument("--kd_temp",        type=float, default=4.0)
    p.add_argument("--ema_decay",      type=float, default=0.999)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="cuda")
    return p.parse_args()


def get_loaders(args):
    train_tf, test_tf = get_tinyimagenet_transforms()
    trainset      = TinyImageNetDataset(args.data_dir, "train", train_tf)
    testset       = TinyImageNetDataset(args.data_dir, "val",   test_tf)
    trainset_eval = TinyImageNetDataset(args.data_dir, "train", test_tf)

    train_labels = torch.tensor([s[1] for s in trainset.samples])
    test_labels  = torch.tensor([s[1] for s in testset.samples])
    fs           = set(args.forget_classes)

    keep_idx        = [i for i,l in enumerate(train_labels) if l.item() not in fs]
    forget_idx      = [i for i,l in enumerate(train_labels) if l.item() in fs]
    test_keep_idx   = [i for i,l in enumerate(test_labels)  if l.item() not in fs]
    test_forget_idx = [i for i,l in enumerate(test_labels)  if l.item() in fs]

    bs = args.batch_size
    keep_loader        = DataLoader(Subset(trainset, keep_idx),
                                    bs, shuffle=True,  num_workers=0)
    forget_loader      = DataLoader(Subset(trainset, forget_idx),
                                    bs, shuffle=True,  num_workers=0)
    test_keep_loader   = DataLoader(Subset(testset, test_keep_idx),
                                    256, shuffle=False, num_workers=0)
    test_forget_loader = DataLoader(Subset(testset, test_forget_idx),
                                    256, shuffle=False, num_workers=0)
    train_keep_eval    = DataLoader(Subset(trainset_eval, keep_idx),
                                    256, shuffle=False, num_workers=0)

    return (keep_loader, forget_loader,
            test_keep_loader, test_forget_loader,
            train_keep_eval, keep_idx)


def compute_prototypes(teacher, keep_loader, device):
    print("Computing prototypes...", flush=True)
    teacher.eval()
    hook = FeatureHook(teacher.layer4)

    with torch.no_grad():
        x, _ = next(iter(keep_loader))
        teacher(x[:1].to(device))
        fd = hook.features.reshape(1, -1).size(1)

    cn = torch.zeros(NUM_CLASSES, fd, dtype=torch.float64)
    cd = torch.zeros(NUM_CLASSES,     dtype=torch.float64)
    gn = torch.zeros(fd,              dtype=torch.float64)
    gd = torch.tensor(0.,             dtype=torch.float64)

    with torch.no_grad():
        for i, (x, y) in enumerate(
                tqdm(keep_loader, desc="  Prototypes", leave=False)):
            x      = x.to(device)
            logits = teacher(x)
            feat   = hook.features.reshape(x.size(0), -1).cpu().double()
            conf   = F.softmax(logits, dim=1).max(1)[0].cpu().double()
            w      = feat * conf.unsqueeze(1)
            yl     = y.long()
            cn.scatter_add_(0, yl.unsqueeze(1).expand(-1, fd), w)
            cd.scatter_add_(0, yl, conf)
            gn += w.sum(0); gd += conf.sum()
            del x, logits, feat, conf, w, yl
            if i % 50 == 0: gc.collect()

    hook.close()
    protos = {c: F.normalize((cn[c]/cd[c]).float().to(device), dim=0)
              for c in range(NUM_CLASSES) if cd[c] > 0}
    gp_vec = F.normalize((gn/gd).float().to(device), dim=0)
    del cn, cd, gn, gd; gc.collect(); torch.cuda.empty_cache()
    print(f"  {len(protos)} prototypes computed.", flush=True)
    return protos, gp_vec


def train_stage1(args, keep_loader, device):
    print("\nStage 1...", flush=True)
    model  = tiny_resnet18(NUM_CLASSES).to(device)
    opt    = torch.optim.SGD(model.parameters(), lr=args.lr_stage1,
                             momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, args.warmup_epochs, args.stage1_epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.stage1_epochs + 1):
        model.train(); cor = tot = tloss = 0
        for x, y in tqdm(keep_loader,
                         desc=f"  S1 {epoch}/{args.stage1_epochs}",
                         leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss   = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tloss += loss.item()
            with torch.no_grad():
                cor += (logits.argmax(1)==y).sum().item()
                tot += y.size(0)
        sch.step()
        print(f"  S1 {epoch}/{args.stage1_epochs}  "
              f"loss={tloss/len(keep_loader):.4f}  "
              f"acc={100*cor/tot:.2f}%", flush=True)
    return model


def train_stage2(args, student, teacher, keep_loader, forget_loader,
                 prototypes, global_proto, val_loader, device):
    print("\nStage 2...", flush=True)
    teacher.eval()
    t_hook = FeatureHook(teacher.layer4)
    s_hook = FeatureHook(student.layer4)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 64, 64, device=device)
        teacher(dummy)
        tch, th, tw = (t_hook.features.size(1),
                       t_hook.features.size(2),
                       t_hook.features.size(3))
        student(dummy)
        sch_dim = s_hook.features.size(1)
        del dummy

    proj  = Projection(sch_dim, tch).to(device)
    opt_s = torch.optim.SGD(
        list(student.parameters()) + list(proj.parameters()),
        lr=args.lr_stage2, momentum=0.9, weight_decay=5e-4)
    sch_s  = make_cosine_scheduler(opt_s, args.warmup_epochs, args.stage2_epochs)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(student, decay=args.ema_decay)

    best_acc = 0.; best_student = None; best_ema = None

    for epoch in range(1, args.stage2_epochs + 1):
        student.train()
        k_it  = iter(keep_loader)
        f_it  = iter(forget_loader)
        steps = max(len(keep_loader), len(forget_loader))
        cp    = args.lambda_push * (epoch / args.stage2_epochs)
        pbar  = tqdm(range(steps),
                     desc=f"  S2 {epoch}/{args.stage2_epochs}",
                     leave=False)

        for _ in pbar:
            try:    xk, yk = next(k_it)
            except: k_it = iter(keep_loader); xk, yk = next(k_it)
            try:    xf, yf = next(f_it)
            except: f_it = iter(forget_loader); xf, yf = next(f_it)

            xk, yk = xk.to(device), yk.to(device)
            xf, yf = xf.to(device), yf.to(device)
            xmix, ya, yb, lam = mixup_batch(xk, yk)

            opt_s.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    tlk    = teacher(xk)
                    tlf    = teacher(xf)
                    conf_f = F.softmax(tlf, dim=1).max(1)[0]

                slk    = student(xk); sfeatk = s_hook.features
                slmix  = student(xmix)
                slf    = student(xf)
                sf     = F.normalize(
                    proj(s_hook.features).view(xf.size(0), -1), dim=1)

                loss_kd   = kd_loss(slk, tlk, args.kd_temp)
                loss_ce   = mixup_criterion(slmix, ya, yb, lam)
                loss_uc   = uniform_confidence_loss(slf, NUM_CLASSES)

                kf        = F.normalize(
                    proj(sfeatk).view(xk.size(0), -1), dim=1)
                pk        = torch.stack(
                    [prototypes.get(int(c), global_proto) for c in yk])
                pf        = torch.stack(
                    [prototypes.get(int(c), global_proto) for c in yf])
                loss_pull = -F.cosine_similarity(kf, pk, dim=1).mean()
                sim       = F.cosine_similarity(sf, pf, dim=1)
                loss_push = (F.relu(sim) * conf_f).mean()

                total = (args.lambda_kd   * loss_kd   +
                         args.lambda_ce   * loss_ce   +
                         args.lambda_uc   * loss_uc   +
                         args.lambda_pull * loss_pull +
                         cp               * loss_push)

            scaler.scale(total).backward()
            scaler.unscale_(opt_s)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt_s); scaler.update()
            ema.update(student)
            pbar.set_postfix(loss=f"{total.item():.3f}")

        sch_s.step()
        acc = quick_accuracy(ema.eval_model(), val_loader, device)
        print(f"  S2 {epoch}/{args.stage2_epochs}  "
              f"EMA={acc:.2f}%  push={cp:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_student = copy.deepcopy(student.state_dict())
            best_ema     = copy.deepcopy(ema.shadow.state_dict())

        if epoch % 10 == 0 or epoch == args.stage2_epochs:
            torch.save({"epoch": epoch,
                        "student": student.state_dict(),
                        "ema": ema.shadow.state_dict()},
                       os.path.join(args.results_dir,
                                    f"stage2_epoch_{epoch}.pth"))

    t_hook.close(); s_hook.close()
    if best_student:
        student.load_state_dict(best_student)
        ema.shadow.load_state_dict(best_ema)
    print(f"  Best EMA val acc: {best_acc:.2f}%", flush=True)
    return student, ema.shadow


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)
    t_start = time.time()

    print(f"PGFU — TinyImageNet | Forget: {args.forget_classes} | Seed: {args.seed}")

    (keep_loader, forget_loader, test_keep_loader,
     test_forget_loader, train_keep_eval, keep_idx) = get_loaders(args)

    teacher = load_teacher(args.teacher_path, "resnet34", NUM_CLASSES, device)

    proto_path = os.path.join(args.results_dir, "prototypes.pt")
    if os.path.exists(proto_path):
        print("Loading cached prototypes...", flush=True)
        saved        = torch.load(proto_path, map_location=device)
        prototypes   = saved["prototypes"]
        global_proto = saved["global_proto"]
    else:
        prototypes, global_proto = compute_prototypes(teacher, keep_loader, device)
        torch.save({"prototypes": prototypes, "global_proto": global_proto},
                   proto_path)

    teacher.cpu(); gc.collect(); torch.cuda.empty_cache()

    s1_path = os.path.join(args.results_dir, "stage1.pth")
    if os.path.exists(s1_path):
        print("Stage 1 cached.", flush=True)
        student = tiny_resnet18(NUM_CLASSES).to(device)
        student.load_state_dict(torch.load(s1_path, map_location=device))
    else:
        student = train_stage1(args, keep_loader, device)
        torch.save(student.state_dict(), s1_path)

    gc.collect(); torch.cuda.empty_cache()
    teacher = teacher.to(device)

    student, ema_student = train_stage2(
        args, student, teacher, keep_loader, forget_loader,
        prototypes, global_proto, test_keep_loader, device)
    del teacher; gc.collect(); torch.cuda.empty_cache()

    torch.save(student.state_dict(),
               os.path.join(args.results_dir, "pgfu_student_final.pth"))
    torch.save(ema_student.state_dict(),
               os.path.join(args.results_dir, "pgfu_ema_final.pth"))

    keep_acc   = compute_accuracy(ema_student, test_keep_loader,
                                   args.forget_classes, False, device)
    forget_acc = compute_accuracy(ema_student, test_forget_loader,
                                   args.forget_classes, True,  device)
    mia_loss   = compute_mia(ema_student, forget_loader,
                              test_forget_loader, "loss", device)
    mia_conf   = compute_mia(ema_student, forget_loader,
                              test_forget_loader, "conf", device)
    elapsed    = (time.time() - t_start) / 60

    results = {"keep": keep_acc, "forget": forget_acc,
               "mia_loss": mia_loss, "mia_conf": mia_conf,
               "train_time_min": elapsed, "seed": args.seed}
    with open(os.path.join(args.results_dir, "pgfu_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Keep={keep_acc:.2f}%  Forget={forget_acc:.2f}%  "
          f"MIA={mia_loss:.4f}  Time={elapsed:.1f}min")


if __name__ == "__main__":
    main()
