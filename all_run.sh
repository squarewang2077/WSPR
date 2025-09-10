#!/usr/bin/env bash
set -e

DATA_ROOT=./dataset
OUT_DIR=./model_zoo/trained_model
mkdir -p "$OUT_DIR"

DATASETS=("cifar10" "cifar100" "tinyimagenet")
ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16" "densenet121" "mobilenet_v3_large" "efficientnet_b0" "vit_b_16")

for ds in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    OUT="$OUT_DIR/${arch}_${ds}.pth"
    if [ -f "$OUT" ]; then
      echo "[skip] $OUT"
      continue
    fi
    # batch size：tiny 用 64，其他 128；按需改
    if [ "$ds" = "tinyimagenet" ]; then BS=64; else BS=128; fi

    echo "[run] $arch on $ds"
    python fit_classifiers.py \
      --dataset "$ds" --data_root "$DATA_ROOT" \
      --arch "$arch" --epochs 20 \
      --batch_size "$BS" --lr 0.01 --weight_decay 5e-4 \
      --label_smoothing 0.0 --pretrained true \
      --device cuda --seed 42 \
      --out "$OUT"
  done
done

echo "[done] all combos."
