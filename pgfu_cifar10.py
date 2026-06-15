"""
pgfu_cifar10.py
===============
PGFU Unlearning — CIFAR-10

Usage:
    python pgfu_cifar10.py \
        --teacher_path checkpoints/teacher_cifar10.pth \
        --results_dir  results/cifar10 \
        --forget_classes 0 1 \
        --seed 42
"""

import argparse, gc, json, os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models.resnet import cifar_resnet18, cifar_resnet34, Projection, StrongCritic, load_teacher
from utils import (ModelEMA, FeatureHook, Cutout, mixup_batch, mixup_criterion,
                   make_cosine_scheduler, kd_loss, uniform_confidence_loss,
                   gradient_penalty, quick_accuracy, set_seed)
from evaluate import compute_accuracy, compute_mia


NUM_CLASSES = 10
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
_MEAN_T = torch.tensor(MEAN).view(3,1,1)
_STD_T  = torch.tensor(STD).view(3,1,1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path",   type=str,   required=True)
    p.add_argument("--results_dir",    type=str,   default="results/cifar10")
    p.add_argument("--data_dir",       type=str,   default="./data")
    p.add_argument("--forget_classes", type=int,   nargs="+", default=[0, 1])
    p.add_argument("--stage1_epochs",  type=int,   default=25)
    p.add_argument("--stage2_epochs",  type=int,   default=40)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr_stage1",      type=float, default=0.05)
    p.add_argument("--lr_stage2",      type=float, default=0.01)
    p.add_argument("--lr_critic",      type=float, default=5e-5)
    p.add_argument("--lambda_kd",      type=float, default=1.0)
    p.add_argument("--lambda_ce",      type=float, default=1.0)
    p.add_argument("--lambda_uc",      type=float, default=0.005)
    p.add_argument("--lambda_pull",    type=float, default=0.01)
    p.add_argument("--lambda_push",    type=float, default=0.15)
    p.add_argument("--lambda_adv",     type=float, default=0.0002)
    p.add_argument("--gp_weight",      type=float, default=5.0)
    p.add_argument("--kd_temp",        type=float, default=4.0)
    p.add_argument("--ema_decay",      type=float, default=0.999)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="cuda")
    return p.parse_args()


def get_loaders(args):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        Cutout(n_holes=1, length=8),
    ])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    trainset = torchvision.datasets.CIFAR10(args.data_dir, True,  download=True, transform=tf_train)
    testset  = torchvision.datasets.CIFAR10(args.data_dir, False, download=True, transform=tf_test)

    train_labels = torch.tensor(trainset.targets)
    test_labels  = torch.tensor(testset.targets)
    fs           = set(args.forget_classes)

    keep_idx        = [i for i,l in enumerate(train_labels) if l.item() not in fs]
    forget_idx      = [i for i,l in enumerate(train_labels) if l.item() in fs]
    test_keep_idx   = [i for i,l in enumerate(test_labels)  if l.item() not in fs]
    test_forget_idx = [i for i,l in enumerate(test_labels)  if l.item() in fs]

    bs = args.batch_size
    return (DataLoader(Subset(trainset, keep_idx),   bs, shuffle=True,  num_workers=0),
            DataLoader(Subset(trainset, forget_idx), bs, shuffle=True,  num_workers=0),
            DataLoader(Subset(testset, test_keep_idx),   256, shuffle=False, num_workers=0),
            DataLoader(Subset(testset, test_forget_idx), 256, shuffle=False, num_workers=0),
            keep_idx, train_labels, trainset)


def compute_prototypes(teacher, keep_idx, trainset_raw, device):
    print("Computing prototypes...", flush=True)
    teacher.eval()
    hook = FeatureHook(teacher.layer4)
    raw  = trainset_raw.data
    labs = np.array(trainset_raw.targets)
    imgs = raw[keep_idx]; lbs = labs[keep_idx]; N = len(imgs)

    with torch.no_grad():
        s = (torch.from_numpy(imgs[:1].copy()).float()/255.).permute(0,3,1,2)
        s = (s - _MEAN_T) / _STD_T; teacher(s.to(device))
        fd = hook.features.reshape(1,-1).size(1)

    cn = torch.zeros(NUM_CLASSES, fd, dtype=torch.float64)
    cd = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    gn = torch.zeros(fd, dtype=torch.float64)
    gd = torch.tensor(0., dtype=torch.float64)
    MINI = 32

    with torch.no_grad():
        for bi in tqdm(range((N+MINI-1)//MINI), desc="  Prototypes", leave=False):
            s = bi*MINI; e = min(s+MINI, N)
            x = (torch.from_numpy(imgs[s:e].copy()).float()/255.).permute(0,3,1,2)
            x = (x - _MEAN_T)/_STD_T
            l = torch.from_numpy(lbs[s:e].copy()).long()
            logits = teacher(x.to(device))
            feat   = hook.features.reshape(e-s,-1).cpu().double()
            conf   = F.softmax(logits,dim=1).max(1)[0].cpu().double()
            w      = feat * conf.unsqueeze(1)
            cn.scatter_add_(0, l.unsqueeze(1).expand(-1,fd), w)
            cd.scatter_add_(0, l, conf)
            gn += w.sum(0); gd += conf.sum()
            del x, l, logits, feat, conf, w

    hook.close()
    protos = {c: F.normalize((cn[c]/cd[c]).float().to(device), dim=0)
              for c in range(NUM_CLASSES) if cd[c] > 0}
    gp_vec = F.normalize((gn/gd).float().to(device), dim=0)
    del cn, cd, gn, gd; gc.collect(); torch.cuda.empty_cache()
    return protos, gp_vec


def train_stage1(args, keep_loader, device):
    print("\nStage 1...", flush=True)
    model  = cifar_resnet18(NUM_CLASSES).to(device)
    opt    = torch.optim.SGD(model.parameters(), lr=args.lr_stage1, momentum=0.9, weight_decay=5e-4)
    sch    = make_cosine_scheduler(opt, args.warmup_epochs, args.stage1_epochs)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, args.stage1_epochs+1):
        model.train(); cor = tot = 0
        for x, y in tqdm(keep_loader, desc=f"  S1 {epoch}/{args.stage1_epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss   = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                cor += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
        sch.step()
        print(f"  S1 {epoch}/{args.stage1_epochs}  acc={100*cor/tot:.2f}%", flush=True)
    return model


def train_stage2(args, student, teacher, keep_loader, forget_loader,
                 prototypes, global_proto, val_loader, device):
    print("\nStage 2...", flush=True)
    teacher.eval()
    t_hook = FeatureHook(teacher.layer4)
    s_hook = FeatureHook(student.layer4)

    with torch.no_grad():
        dummy = torch.randn(1,3,32,32,device=device)
        teacher(dummy); tch,th,tw = t_hook.features.size(1), t_hook.features.size(2), t_hook.features.size(3)
        student(dummy); sch_dim = s_hook.features.size(1); del dummy

    proj   = Projection(sch_dim, tch).to(device)
    critic = StrongCritic(tch*th*tw).to(device)
    opt_s  = torch.optim.SGD(list(student.parameters())+list(proj.parameters()),
                              lr=args.lr_stage2, momentum=0.9, weight_decay=5e-4)
    sch_s  = make_cosine_scheduler(opt_s, args.warmup_epochs, args.stage2_epochs)
    opt_c  = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    scaler = torch.cuda.amp.GradScaler()
    ema    = ModelEMA(student, decay=args.ema_decay)
    best_acc = 0.; best_student = None; best_ema = None

    for epoch in range(1, args.stage2_epochs+1):
        student.train()
        k_it = iter(keep_loader); f_it = iter(forget_loader)
        steps = max(len(keep_loader), len(forget_loader))
        cp = args.lambda_push * (epoch / args.stage2_epochs)

        for _ in tqdm(range(steps), desc=f"  S2 {epoch}/{args.stage2_epochs}", leave=False):
            try:    xk,yk = next(k_it)
            except: k_it = iter(keep_loader); xk,yk = next(k_it)
            try:    xf,yf = next(f_it)
            except: f_it = iter(forget_loader); xf,yf = next(f_it)
            xk,yk = xk.to(device),yk.to(device)
            xf,yf = xf.to(device),yf.to(device)
            xmix,ya,yb,lam = mixup_batch(xk,yk)

            with torch.no_grad():
                tlf = teacher(xf)
                tf_flat = F.normalize(t_hook.features.view(xf.size(0),-1).detach(),dim=1)
                conf_f  = F.softmax(tlf,dim=1).max(1)[0]
                student(xf)
                sf_flat = F.normalize(proj(s_hook.features).view(xf.size(0),-1),dim=1)

            opt_c.zero_grad()
            lc = critic(sf_flat).mean() - critic(tf_flat).mean()
            gp_val = gradient_penalty(critic, tf_flat.detach(), sf_flat.detach(), device)
            (lc + args.gp_weight * gp_val).backward(); opt_c.step()

            opt_s.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad(): tlk = teacher(xk)
                slk = student(xk); sfeatk = s_hook.features
                slmix = student(xmix)
                slf   = student(xf)
                sf    = F.normalize(proj(s_hook.features).view(xf.size(0),-1),dim=1)
                loss_kd   = kd_loss(slk, tlk, args.kd_temp)
                loss_ce   = mixup_criterion(slmix, ya, yb, lam)
                loss_uc   = uniform_confidence_loss(slf, NUM_CLASSES)
                loss_adv  = -critic(sf).mean()
                kf        = F.normalize(proj(sfeatk).view(xk.size(0),-1),dim=1)
                pk        = torch.stack([prototypes.get(int(c),global_proto) for c in yk])
                pf        = torch.stack([prototypes.get(int(c),global_proto) for c in yf])
                loss_pull = -F.cosine_similarity(kf,pk,dim=1).mean()
                sim       = F.cosine_similarity(sf,pf,dim=1)
                loss_push = (F.relu(sim)*conf_f).mean()
                total = (args.lambda_kd*loss_kd + args.lambda_ce*loss_ce +
                         args.lambda_uc*loss_uc + args.lambda_adv*loss_adv +
                         args.lambda_pull*loss_pull + cp*loss_push)

            scaler.scale(total).backward()
            scaler.unscale_(opt_s)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt_s); scaler.update(); ema.update(student)

        sch_s.step()
        acc = quick_accuracy(ema.eval_model(), val_loader, device)
        print(f"  S2 {epoch}/{args.stage2_epochs}  EMA={acc:.2f}%  push={cp:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_student = copy.deepcopy(student.state_dict())
            best_ema     = copy.deepcopy(ema.shadow.state_dict())
        if epoch % 10 == 0 or epoch == args.stage2_epochs:
            torch.save({"epoch":epoch,"student":student.state_dict(),"ema":ema.shadow.state_dict()},
                       os.path.join(args.results_dir, f"stage2_epoch_{epoch}.pth"))

    t_hook.close(); s_hook.close()
    if best_student: student.load_state_dict(best_student); ema.shadow.load_state_dict(best_ema)
    print(f"  Best EMA: {best_acc:.2f}%", flush=True)
    return student, ema.shadow


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)
    t_start = time.time()
    print(f"PGFU — CIFAR-10 | Forget: {args.forget_classes} | Seed: {args.seed}")

    (keep_loader, forget_loader, test_keep_loader,
     test_forget_loader, keep_idx, train_labels, trainset_raw) = get_loaders(args)

    teacher = load_teacher(args.teacher_path, "resnet34", NUM_CLASSES, device)

    proto_path = os.path.join(args.results_dir, "prototypes.pt")
    if os.path.exists(proto_path):
        saved = torch.load(proto_path, map_location=device)
        prototypes = saved["prototypes"]; global_proto = saved["global_proto"]
    else:
        prototypes, global_proto = compute_prototypes(teacher, keep_idx, trainset_raw, device)
        torch.save({"prototypes": prototypes, "global_proto": global_proto}, proto_path)

    teacher.cpu(); gc.collect(); torch.cuda.empty_cache()

    s1_path = os.path.join(args.results_dir, "stage1.pth")
    if os.path.exists(s1_path):
        student = cifar_resnet18(NUM_CLASSES).to(device)
        student.load_state_dict(torch.load(s1_path, map_location=device))
    else:
        student = train_stage1(args, keep_loader, device)
        torch.save(student.state_dict(), s1_path)

    teacher = teacher.to(device)
    student, ema_student = train_stage2(
        args, student, teacher, keep_loader, forget_loader,
        prototypes, global_proto, test_keep_loader, device)
    del teacher; gc.collect(); torch.cuda.empty_cache()

    torch.save(student.state_dict(),     os.path.join(args.results_dir, "pgfu_student_final.pth"))
    torch.save(ema_student.state_dict(), os.path.join(args.results_dir, "pgfu_ema_final.pth"))

    keep_acc   = compute_accuracy(ema_student, test_keep_loader,   args.forget_classes, False, device)
    forget_acc = compute_accuracy(ema_student, test_forget_loader, args.forget_classes, True,  device)
    mia_loss   = compute_mia(ema_student, forget_loader, test_forget_loader, "loss", device)
    mia_conf   = compute_mia(ema_student, forget_loader, test_forget_loader, "conf", device)
    elapsed    = (time.time()-t_start)/60

    results = {"keep":keep_acc,"forget":forget_acc,"mia_loss":mia_loss,
               "mia_conf":mia_conf,"train_time_min":elapsed,"seed":args.seed}
    with open(os.path.join(args.results_dir,"pgfu_metrics.json"),"w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Keep={keep_acc:.2f}%  Forget={forget_acc:.2f}%  "
          f"MIA={mia_loss:.4f}  Time={elapsed:.1f}min")

if __name__ == "__main__":
    main()
