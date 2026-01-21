#!/bin/bash

# Script to retrain ViT-B-16 on CIFAR10/100 and TinyImageNet with correct 224x224 input size
# This fixes the dimension mismatch issue by resizing datasets to 3x224x224

set -e  # Exit on error

# Configuration
GPU=${1:-0}  # Default GPU 0, can override with first argument
EPOCHS=50
BATCH_SIZE=128
LR=0.01
ARCH="vit_b_16"

# Datasets to train on
DATASETS=("cifar10" "cifar100" "tinyimagenet")

# Create directories
mkdir -p ./model_zoo/trained_model
mkdir -p ./model_zoo/trained_model/training_log_info

echo "======================================"
echo "ViT-B-16 Retraining Script"
echo "======================================"
echo "GPU: $GPU"
echo "Architecture: $ARCH"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Image size: 224x224 (resized from native dataset sizes)"
echo "Datasets: ${DATASETS[@]}"
echo "======================================"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

# Train on each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "======================================"
    echo "Training: $ARCH on $dataset"
    echo "======================================"
    echo "Start time: $(date)"

    # Output paths
    OUTPUT_PATH="./model_zoo/trained_model/${ARCH}_${dataset}.pth"
    LOG_FILE="./model_zoo/trained_model/training_log_info/${ARCH}_${dataset}.log"

    # Run training with 224x224 image size (ViT requirement)
    # Note: NOT passing --img_size lets the script default to 224 for pretrained models
    # Or we can explicitly pass --img_size 224
    CUDA_VISIBLE_DEVICES=$GPU python fit_classifiers.py \
        --dataset "$dataset" \
        --arch "$ARCH" \
        --pretrained true \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --device cuda \
        --out "$OUTPUT_PATH" \
        2>&1 | tee "$LOG_FILE"

    # Check if training succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ SUCCESS: $ARCH on $dataset"
        echo "  Model saved to: $OUTPUT_PATH"
        echo "  Log saved to: $LOG_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ FAILED: $ARCH on $dataset"
        echo "  Check log: $LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo "End time: $(date)"
    echo ""
done

# Summary
echo ""
echo "======================================"
echo "Training Summary"
echo "======================================"
echo "Total datasets: ${#DATASETS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "======================================"
echo ""

# List trained models
echo "Trained ViT models:"
ls -lh ./model_zoo/trained_model/vit_b_16_*.pth 2>/dev/null || echo "No models found"
echo ""
echo "Training logs:"
ls -lh ./model_zoo/trained_model/training_log_info/vit_b_16_*.log 2>/dev/null || echo "No logs found"
