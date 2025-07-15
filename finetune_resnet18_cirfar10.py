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
learning_rate = 0.03  # 对微调建议用较小的学习率
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
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pretrained model and modify for CIFAR-10
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)  # 加载预训练
for param in net.parameters():
    param.requires_grad = False  # 冻结所有层

# 替换最后一层全连接层为 CIFAR-10 分类
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 10)  # 新的未预训练层
for param in net.fc.parameters():
    param.requires_grad = True  # 解冻最后一层参数
net = net.to(device)
# Loss and optimizer (只优化最后一层参数)
criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the last fully connected layer (net.fc) for fine-tuning
optimizer = optim.Adam(net.fc.parameters(), lr=learning_rate, weight_decay=5e-4)
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
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    results.append({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })  
    # For PyTorch 1.1.0+ call scheduler.step() after optimizer.step()
    scheduler.step()

# Save model
model_path = os.path.join(model_dir, "resnet18_finetuned_cifar10.pth")
torch.save(net.state_dict(), model_path)
# Save results to CSV
df = pd.DataFrame(results)
csv_path = os.path.join(log_dir, "resnet18_cifar10_finetuning.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False)
