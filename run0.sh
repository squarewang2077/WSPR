#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0


# for num_modes in 20
# do
#     for gamma in  0.0157 0.0314 0.0627
#     do
#         echo "Running with gamma=$gamma"
#         echo "Running with num_modes=$num_modes"
#         python fit_gmm.py \
#         --num_modes $num_modes \
#         --gamma $gamma \
#         --norm linf 
#     done
# done

# for num_modes in 20
# do
#     for gamma in  0.0157 0.0314 0.0627
#     do
#         echo "Running with gamma=$gamma"
#         echo "Running with num_modes=$num_modes"
#         python fit_gmm.py \
#         --num_modes $num_modes \
#         --gamma $gamma \
#         --norm linf \
#         --xdep
#     done
# done

python fit_gmm2.py --dataset cifar10 --arch resnet18 --clf_ckpt ./model_zoo/trained_model/resnet18_cifar10.pth --K 7 --condition xy --use_decoder --decoder_backend bicubic