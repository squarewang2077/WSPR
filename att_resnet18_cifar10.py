import os
import torch
import torchvision
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from attacks import nattack, WithIndex

# Directories
model_dir = "model_zoo/trained_model/ResNets"
log_dir = "log/training_results"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Hyperparameters
num_epochs = 20
batch_size = 512
learning_rate = 0.001
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Datasets and loaders
trainset = WithIndex(torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = WithIndex(torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Model
# net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)  
net = torchvision.models.resnet18(weights=None, num_classes=10)
net = net.to(device)


# Load model
model_path = os.path.join(model_dir, "resnet18_cifar10.pth")
net.load_state_dict(torch.load(model_path))

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


out = nattack(
    net, testloader,
    fraction=0.1, # ≈ 10% of samples
    device=None,
    save_perturbations=False,
    N_noise=300, sigma=0.1, \
    att_type="infty", epsi=0.031, \
    N_step=500, step_size=0.02, reward_scaling=0.5, \
    img_mean = [0.4914, 0.4822, 0.4465], img_std = [0.2023, 0.1994, 0.2010]
)

print(out['succeeded_indices'][:20])   # dataset indices where the attack succeeded
print(out['counts'])
