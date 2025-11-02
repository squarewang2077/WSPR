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

python fit_gmm2.py --config resnet18_on_cifar10_linf_K3
# python fit_gmm2.py --config resnet18_on_cifar10_linf_K7
# python fit_gmm2.py --config resnet18_on_cifar10_linf_K12
# python fit_gmm2.py --config x_dependent_test_2