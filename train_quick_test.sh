#!/bin/bash

# Quick test script - trains ResNet18 on CIFAR-10 and CIFAR-100
# Useful for testing before running the full training suite

echo "======================================"
echo "Quick Training Test"
echo "======================================"
echo "This will train:"
echo "  - ResNet18 on CIFAR-10 (32x32)"
echo "  - ResNet18 on CIFAR-100 (32x32)"
echo ""
echo "With reduced epochs for quick testing"
echo "======================================"
echo ""

# Configuration
GPU=${1:-0}
EPOCHS=10
BATCH_SIZE=128

mkdir -p ./model_zoo/trained_model
mkdir -p ./logs

echo "GPU: $GPU"
echo "Epochs: $EPOCHS (reduced for testing)"
echo ""

# Train ResNet18 on CIFAR-10
echo "[1/2] Training ResNet18 on CIFAR-10..."
CUDA_VISIBLE_DEVICES=$GPU python fit_classifiers.py \
    --dataset cifar10 \
    --arch resnet18 \
    --pretrained true \
    --img_size 32 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.01 \
    --device cuda

echo ""

# Train ResNet18 on CIFAR-100
echo "[2/2] Training ResNet18 on CIFAR-100..."
CUDA_VISIBLE_DEVICES=$GPU python fit_classifiers.py \
    --dataset cifar100 \
    --arch resnet18 \
    --pretrained true \
    --img_size 32 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.01 \
    --device cuda

echo ""
echo "======================================"
echo "Quick test complete!"
echo "======================================"
echo "Trained models:"
ls -lh ./model_zoo/trained_model/resnet18_*.pth
