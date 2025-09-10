#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified trainer for CIFAR-10 / CIFAR-100 / TinyImageNet with multiple backbones.
Supports ImageNet-pretrained weights to accelerate training.

Architectures:
  - resnet18, resnet50, wide_resnet50_2
  - vgg16
  - densenet121
  - mobilenet_v3_large
  - efficientnet_b0
  - vit_b_16

Datasets:
  - cifar10, cifar100, tinyimagenet (val folder layout from the official release)

Notes:
  * With --pretrained (default=True), images are resized to 224×224 and normalized with ImageNet stats,
    so we can load torchvision's ImageNet weights directly.
  * With --pretrained false, we use dataset-native sizes & stats, and train from scratch.

Save:
  * The best checkpoint by validation accuracy is saved to:
        ./model_zoo/trained_model/<arch>_<dataset>.pth
    unless you pass a custom --out.
"""

import os
import argparse
import time
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.datasets as dsets
from tqdm import tqdm

# ------------------------------ Utilities ------------------------------

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

CIFAR10_MEAN, CIFAR10_STD   = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
TINY_MEAN, TINY_STD         = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

def set_seed(seed: int = 42):
    """Make training as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_norm_stats(dataset: str, use_imnet_stats: bool):
    """Return (mean, std) for normalization."""
    if use_imnet_stats:
        return IMNET_MEAN, IMNET_STD
    dataset = dataset.lower()
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if dataset == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    if dataset == "tinyimagenet":
        return TINY_MEAN, TINY_STD
    raise ValueError(f"Unknown dataset {dataset}")


def get_default_img_size(arch: str, pretrained: bool, dataset: str) -> int:
    """
    Decide input size:
      - if pretrained: 224 (to match ImageNet weights) for all listed models
      - else: native dataset sizes (CIFAR: 32; TinyImageNet: 64)
    """
    if pretrained:
        return 224
    if dataset == "tinyimagenet":
        return 64
    return 32  # CIFAR10/100


def get_dataset(
    name: str,
    root: str,
    train: bool,
    img_size: int,
    use_imnet_stats: bool
):
    """
    Build dataset + num_classes based on name, with transforms depending on
    `img_size` and normalization stats choice.
    """
    name = name.lower()
    mean, std = get_norm_stats(name, use_imnet_stats)

    if train:
        tf = T.Compose([
            T.Resize(img_size),
            T.RandomCrop(img_size, padding=int(0.125*img_size)) if img_size >= 64 else T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    else:
        tf = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    if name == "cifar10":
        ds = dsets.CIFAR10(root=root, train=train, download=True, transform=tf)
        num_classes = 10
    elif name == "cifar100":
        ds = dsets.CIFAR100(root=root, train=train, download=True, transform=tf)
        num_classes = 100
    elif name == "tinyimagenet":
        split = "train" if train else "val"
        base = os.path.join(root, "tiny-imagenet-200", split)
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"TinyImageNet folder not found at {base}.\n"
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to <root>/tiny-imagenet-200/"
            )
        ds = dsets.ImageFolder(base, transform=tf)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset {name}")

    return ds, num_classes


def replace_classifier(model: nn.Module, arch: str, num_classes: int):
    """Replace the classifier head to match num_classes, keeping pretrained trunk."""
    arch = arch.lower()
    if arch.startswith("resnet") or arch.startswith("wide_resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        new_classifier = list(model.classifier[:-1]) + [nn.Linear(in_features, num_classes)]
        model.classifier = nn.Sequential(*new_classifier)
    elif arch.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch.startswith("mobilenet_v3"):
        in_features = model.classifier[-1].in_features
        new_classifier = list(model.classifier[:-1]) + [nn.Linear(in_features, num_classes)]
        model.classifier = nn.Sequential(*new_classifier)
    elif arch.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        new_classifier = list(model.classifier[:-1]) + [nn.Linear(in_features, num_classes)]
        model.classifier = nn.Sequential(*new_classifier)
    elif arch.startswith("vit_"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unhandled arch for head replacement: {arch}")
    return model


def build_model(arch: str, num_classes: int, device, pretrained: bool):
    """
    Build a torchvision model and (optionally) load ImageNet weights.
    We then replace the classifier head to match `num_classes`.
    """
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
        wide_resnet50_2, Wide_ResNet50_2_Weights,
        vgg16, VGG16_Weights,
        densenet121, DenseNet121_Weights,
        mobilenet_v3_large, MobileNet_V3_Large_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
        vit_b_16, ViT_B_16_Weights
    )

    arch = arch.lower()
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    elif arch == "wide_resnet50_2":
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT if pretrained else None)
    elif arch == "vgg16":
        model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
    elif arch == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
    elif arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    elif arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif arch == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model = replace_classifier(model, arch, num_classes)
    model.to(device)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    """Return accuracy on `loader`."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

# ------------------------------ Training ------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    # Wrap loader with tqdm
    pbar = tqdm(loader, desc=f"Train Epoch [{epoch}/{total_epochs}]" if epoch else "Training", leave=False)
    
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'),
                            dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * y.size(0)
        avg_loss = running_loss / ((pbar.n + 1) * y.size(0))  # average loss

        pbar.set_postfix(loss=f"{avg_loss:.4f}")  # loss on the progress bar

    return running_loss / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", choices=[
        "resnet18","resnet50","wide_resnet50_2",
        "vgg16","densenet121","mobilenet_v3_large","efficientnet_b0",
        "vit_b_16"
    ], default="resnet18")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--pretrained", type=lambda s: s.lower() in ["1","true","yes","y"], default=True,
                    help="Use ImageNet-pretrained weights (and 224×224 inputs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    # If --out is empty, we will save to ./model_zoo/trained_model/<arch>_<dataset>.pth
    ap.add_argument("--out", type=str, default="",
                    help="Optional explicit path to save best checkpoint")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Decide input size + normalization policy
    img_size = get_default_img_size(args.arch, args.pretrained, args.dataset)
    use_imnet_stats = bool(args.pretrained)

    # Build datasets/loaders
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True,  img_size, use_imnet_stats)
    test_set,  _           = get_dataset(args.dataset, args.data_root, False, img_size, use_imnet_stats)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Build model (with or without pretrained weights) + replace head
    model = build_model(args.arch, num_classes, device, pretrained=args.pretrained)

    # Optional DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ---- auto output path (unless user provided one) ----
    if args.out.strip():
        out_path = args.out
    else:
        save_dir = os.path.join(".", "model_zoo", "trained_model")
        os.makedirs(save_dir, exist_ok=True)
        # e.g., resnet18_cifar10.pth
        base = f"{args.arch.lower()}_{args.dataset.lower()}.pth"
        out_path = os.path.join(save_dir, base)
    print(f"[save] best checkpoint will be written to: {out_path}")

    # Train
    best_acc = 0.0
    start = time.time()
    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion,
                                    epoch=ep, total_epochs=args.epochs)

        acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"[{ep:03d}/{args.epochs}] loss={train_loss:.4f}  val_acc={acc*100:.2f}%  lr={scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint
        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "epoch": ep,
                "arch": args.arch,
                "dataset": args.dataset,
                "img_size": img_size,
                "pretrained": args.pretrained,
                "model_state": model.state_dict()
            }
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save(ckpt, out_path)
            print(f"  -> saved best to {out_path}")

    elapsed = time.time() - start
    print(f"Done. Best val acc: {best_acc*100:.2f}%  (time: {elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
