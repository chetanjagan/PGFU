#!/bin/bash
# ============================================================
# scripts/reproduce_all.sh
# Reproduces ALL results from the PGFU paper.
#
# Prerequisites:
#   pip install -r requirements.txt
#   Place teacher checkpoints in ./checkpoints/
#   Place TinyImageNet at ./data/tiny-imagenet-200/
#
# Usage:
#   bash scripts/reproduce_all.sh
# ============================================================

set -e

DEVICE="cuda"
DATA_DIR="./data"
CKPT_DIR="./checkpoints"
RESULTS="./results"

echo "============================================================"
echo "  PGFU Full Reproduction"
echo "============================================================"

# ── CIFAR-10 ─────────────────────────────────────────────────
echo ""
echo "[1/6] PGFU — CIFAR-10 (forget: airplane, automobile)"
python pgfu_cifar10.py \
    --teacher_path  $CKPT_DIR/teacher_cifar10.pth \
    --data_dir      $DATA_DIR \
    --results_dir   $RESULTS/cifar10 \
    --forget_classes 0 1 \
    --seed 42 \
    --device $DEVICE

# ── CIFAR-100 (3 seeds) ───────────────────────────────────────
echo ""
echo "[2/6] PGFU — CIFAR-100 (3 seeds: 42, 123, 456)"
for SEED in 42 123 456; do
    echo "  Seed $SEED..."
    python pgfu_cifar100.py \
        --teacher_path  $CKPT_DIR/teacher_cifar100_resnet34.pth \
        --data_dir      $DATA_DIR \
        --results_dir   $RESULTS/cifar100_seed${SEED} \
        --forget_classes 0 1 2 \
        --seed $SEED \
        --device $DEVICE
done

# ── TinyImageNet (3 seeds) ────────────────────────────────────
echo ""
echo "[3/6] PGFU — TinyImageNet (3 seeds: 42, 123, 456)"
for SEED in 42 123 456; do
    echo "  Seed $SEED..."
    python pgfu_tinyimagenet.py \
        --teacher_path  $CKPT_DIR/best.pth \
        --data_dir      $DATA_DIR/tiny-imagenet-200 \
        --results_dir   $RESULTS/tinyimagenet_seed${SEED} \
        --forget_classes 0 1 2 3 4 5 6 7 8 9 \
        --seed $SEED \
        --device $DEVICE
done

# ── CIFAR-100 Baselines ───────────────────────────────────────
echo ""
echo "[4/6] Baselines — CIFAR-100"
python baselines/run_baselines_cifar100.py \
    --teacher_path  $CKPT_DIR/teacher_cifar100_resnet34.pth \
    --data_dir      $DATA_DIR \
    --results_dir   $RESULTS/cifar100_baselines \
    --forget_classes 0 1 2 \
    --device $DEVICE

# ── Forget-Set Scaling ────────────────────────────────────────
echo ""
echo "[5/6] Forget-Set Scaling — CIFAR-100"
for N_CLASSES in 3 10 50; do
    CLASSES=$(python3 -c "print(' '.join(map(str, range($N_CLASSES))))")
    echo "  Forget $N_CLASSES classes..."
    python pgfu_cifar100.py \
        --teacher_path  $CKPT_DIR/teacher_cifar100_resnet34.pth \
        --data_dir      $DATA_DIR \
        --results_dir   $RESULTS/cifar100_scaling_${N_CLASSES}cls \
        --forget_classes $CLASSES \
        --seed 42 \
        --device $DEVICE
done

# ── Linear Probe ──────────────────────────────────────────────
echo ""
echo "[6/6] Linear Probe Evaluation"
python linear_probe.py \
    --checkpoint          $RESULTS/cifar100_seed42/pgfu_ema_final.pth \
    --retrain_checkpoint  $RESULTS/cifar100_baselines/retrain_model.pth \
    --dataset             cifar100 \
    --data_dir            $DATA_DIR \
    --forget_classes      0 1 2 \
    --output              $RESULTS/cifar100_linear_probe.json \
    --device $DEVICE

echo ""
echo "============================================================"
echo "  All done! Results saved to ./results/"
echo "============================================================"
