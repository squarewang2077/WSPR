import os
import torch
import torchvision
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Directories
model_dir = "model_zoo/trained_model/ResNets"
log_dir = "log/training_results"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Hyperparameters
num_epochs = 20
batch_size = 512
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
# net = torchvision.models.resnet18(weights=None, num_classes=10)
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)  
net = net.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training and evaluation
results = []

def evaluate(loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

for epoch in range(num_epochs):
    net.train()
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_acc = 100. * correct / total
    test_acc = evaluate(testloader)
    results.append({'epoch': epoch+1, 'train_acc': train_acc, 'test_acc': test_acc})
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    scheduler.step()

# Save model
model_path = os.path.join(model_dir, "resnet18_cifar10.pth")
torch.save(net.state_dict(), model_path)

# Save results to CSV
df = pd.DataFrame(results)
csv_path = os.path.join(log_dir, "resnet18_cifar10_training.csv")
df.to_csv(csv_path, index=False)