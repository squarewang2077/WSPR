#!/usr/bin/env bash

# # Enable command logging
# set -x

# # Create log file with timestamp
# LOGFILE="run0_$(date +%Y%m%d_%H%M%S).log"
# exec > >(tee -a "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES=0

# echo "=== Starting run0.sh at $(date) ==="
# echo "Log file: $LOGFILE"


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

# python fit_gmm2.py --config resnet18_on_cifar10_linf_K3
# python fit_gmm2.py --config resnet18_on_cifar10_linf_K7
# python fit_gmm2.py --config resnet18_on_cifar10_linf_K12
# python fit_gmm2.py --config x_dependent_test_2

# python fit_gmm2.py --config resnet18_on_cifar10_linf_4

# python fit_gmm2.py --config resnet18_on_cifar100
# python fit_gmm2.py --config resnet50_on_cifar100
# python fit_gmm2.py --config wrn50_on_cifar100
# python fit_gmm2.py --config vgg16_on_cifar100

# python fit_gmm2.py --config vit_on_cifar10
# python fit_gmm2.py --config vit_on_cifar100
# python fit_gmm2.py --config vit_on_tinyimagenet

python fit_gmm.py --config efficientnet_on_cifar10
python fit_gmm.py --config efficientnet_on_cifar100
