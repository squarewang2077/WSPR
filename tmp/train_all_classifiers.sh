#!/bin/bash

# Script to train all classifier combinations
# Usage: ./train_all_classifiers.sh [OPTIONS]
#
# Options:
#   --gpu N          GPU device to use (default: 0)
#   --epochs N       Number of epochs (default: 50)
#   --batch_size N   Batch size (default: 128)
#   --lr FLOAT       Learning rate (default: 0.01)
#   --quick          Quick mode: only train ResNet18 and ResNet50
#   --datasets LIST  Comma-separated list of datasets (default: cifar10,cifar100,tinyimagenet)
#   --archs LIST     Comma-separated list of architectures (default: all)

# Parse command line arguments
GPU=0
EPOCHS=50
BATCH_SIZE=128
LR=0.01
QUICK_MODE=false
DATASETS="cifar10,cifar100,tinyimagenet"
ARCHS="resnet18,resnet50,wide_resnet50_2,vgg16,densenet121,mobilenet_v3_large,efficientnet_b0,vit_b_16"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            ARCHS="resnet18,resnet50"
            shift
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --archs)
            ARCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert comma-separated lists to arrays
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra ARCH_ARRAY <<< "$ARCHS"

# Create log directory
LOG_DIR="./logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Create model output directory
mkdir -p ./model_zoo/trained_model

echo "======================================"
echo "Training Configuration"
echo "======================================"
echo "GPU: $GPU"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Quick mode: $QUICK_MODE"
echo "Datasets: ${DATASET_ARRAY[@]}"
echo "Architectures: ${ARCH_ARRAY[@]}"
echo "Log directory: $LOG_DIR"
echo "======================================"
echo ""

# Counter for tracking progress
TOTAL_RUNS=$((${#DATASET_ARRAY[@]} * ${#ARCH_ARRAY[@]}))
CURRENT_RUN=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Main training loop
for dataset in "${DATASET_ARRAY[@]}"; do
    # Determine image size based on dataset
    if [ "$dataset" = "tinyimagenet" ]; then
        IMG_SIZE=64
    else
        IMG_SIZE=32  # cifar10 or cifar100
    fi

    for arch in "${ARCH_ARRAY[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo ""
        echo "======================================"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Training: $arch on $dataset"
        echo "======================================"
        echo "Start time: $(date)"

        # Output path
        OUTPUT_PATH="./model_zoo/trained_model/${arch}_${dataset}.pth"
        LOG_FILE="$LOG_DIR/${arch}_${dataset}.log"

        # Run training
        CUDA_VISIBLE_DEVICES=$GPU python fit_classifiers.py \
            --dataset "$dataset" \
            --arch "$arch" \
            --pretrained true \
            --img_size "$IMG_SIZE" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --device cuda \
            --out "$OUTPUT_PATH" \
            2>&1 | tee "$LOG_FILE"

        # Check if training succeeded
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ SUCCESS: $arch on $dataset"
            echo "  Model saved to: $OUTPUT_PATH"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "✗ FAILED: $arch on $dataset"
            echo "  Check log: $LOG_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        echo "End time: $(date)"
        echo ""
    done
done

# Summary
echo ""
echo "======================================"
echo "Training Summary"
echo "======================================"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Logs saved to: $LOG_DIR"
echo "Models saved to: ./model_zoo/trained_model/"
echo "======================================"
echo ""

# List all trained models
echo "Trained models:"
ls -lh ./model_zoo/trained_model/*.pth 2>/dev/null || echo "No models found"
