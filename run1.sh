#!/usr/bin/env bash

# # Enable command logging
# set -x

# # Create log file with timestamp
# LOGFILE="./logs/run1_$(date +%Y%m%d_%H%M%S).log"
# exec > >(tee -a "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES=1

# echo "=== Starting run1.sh at $(date) ==="
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
#     for gamma in 0.25 0.5 1.0
#     do
#         echo "Running with gamma=$gamma"
#         echo "Running with num_modes=$num_modes"
#         python fit_gmm.py \
#         --num_modes $num_modes \
#         --gamma $gamma \
#         --norm l2 \
#         --xdep
#     done
# done

# python fit_gmm2.py --config resnet18_on_cifar10_l2_K3
# python fit_gmm2.py --config resnet18_on_cifar10_l2_K7
# python fit_gmm2.py --config resnet18_on_cifar10_l2_K12
# python fit_gmm2.py --config x_dependent_test_4

# python fit_gmm2.py --config resnet18_on_cifar10_fixed

# python fit_gmm2.py --config resnet18_on_tinyimagenet
# python fit_gmm2.py --config resnet50_on_cifar10
# python fit_gmm2.py --config wrn50_on_cifar10
# python fit_gmm2.py --config vgg16_on_cifar10

# python fit_gmm2.py --config resnet50_on_cifar10
# python fit_gmm2.py --config wrn50_on_cifar10
# python fit_gmm2.py --config vgg16_on_cifar10

# python fit_gmm2.py --config resnet18_on_tinyimagenet_cond_y
python fit_gmm.py --config efficientnet_on_tinyimagenet


