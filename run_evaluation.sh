#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

for model in resnet18
do
    for dataset in cifar10
    do
    python test.py \
        --dataset $dataset \
        --arch $model \
        --clf_ckpt ./model_zoo/trained_model/${model}_${dataset}.pth \
        --epsilon 0.016 \
        --norm_type linf \
        --num_samples 500 \
        --attack_steps 20 \
        --step_size 0.00784 \
        --max_batches 100 \
        --log_dir ./logs
    done
done
