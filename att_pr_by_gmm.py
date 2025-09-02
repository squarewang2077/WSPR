import os
import torch
import torchvision
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from attacks import WithIndex, fit_gmm


def main():
    parser = argparse.ArgumentParser(description='fit the perturbation distribution (GMM)')

    parser.add_argument('--num_modes', type=int, default=1, help='number of modes for GMM')
    parser.add_argument('--cov_type', type=str, default='diagonal', help='covariance type for GMM')
    parser.add_argument('--classifier', type=str, default='ResNet18', help='classifier architecture')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use for training')


    args = parser.parse_args()

    # Directories
    model_dir = "model_zoo/trained_model/ResNets"
    log_dir = "log/training_results"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Hyperparameters
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ### Data Setup ###
    # Data transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Datasets and loaders
    testset = WithIndex(torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    ### Data Setup End ### 

    ### Model Setup ###
    if args.classifier == 'ResNet18':
        net = torchvision.models.resnet18(weights=None, num_classes=10)
        
    net = net.to(device)

    # Load model
    model_path = os.path.join(model_dir, "resnet18_cifar10.pth")
    net.load_state_dict(torch.load(model_path))
    ### Model Setup End ###

    # function for evaluation
    def evaluate(loader):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, _idx in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total

    print("Test Accuracy: {:.2f}%".format(evaluate(testloader)))

    att_pr(net, testloader, args)

if __name__ == '__main__':
    main()
