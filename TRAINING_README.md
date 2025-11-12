# Classifier Training Scripts

This directory contains scripts for training image classifiers on CIFAR-10, CIFAR-100, and TinyImageNet datasets.

## Quick Start

### 1. Test the Setup (Recommended First)
Run a quick test with reduced epochs to verify everything works:

```bash
./train_quick_test.sh 0  # 0 is the GPU ID
```

This trains ResNet18 on CIFAR-10 and CIFAR-100 with just 10 epochs.

### 2. Train All Models
Train all architecture-dataset combinations:

```bash
./train_all_classifiers.sh
```

## Scripts Overview

### `fit_classifiers.py`
The main training script with support for:
- **Architectures**: ResNet18, ResNet50, Wide ResNet50-2, VGG16, DenseNet121, MobileNetV3-Large, EfficientNet-B0, ViT-B/16
- **Datasets**: CIFAR-10, CIFAR-100, TinyImageNet
- **Features**: ImageNet pretrained weights, mixed-precision training, native image sizes

**Example Usage:**
```bash
# Train ResNet18 on CIFAR-100 with pretrained weights and native 32x32 images
python fit_classifiers.py --dataset cifar100 --arch resnet18 --pretrained true --img_size 32

# Train from scratch on TinyImageNet
python fit_classifiers.py --dataset tinyimagenet --arch resnet50 --pretrained false

# Custom configuration
python fit_classifiers.py \
    --dataset cifar100 \
    --arch vit_b_16 \
    --pretrained true \
    --img_size 32 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001
```

### `train_all_classifiers.sh`
Automated script to train all combinations with logging and error handling.

**Usage:**
```bash
# Train all models with default settings
./train_all_classifiers.sh

# Custom GPU and epochs
./train_all_classifiers.sh --gpu 1 --epochs 100

# Quick mode (ResNet18 and ResNet50 only)
./train_all_classifiers.sh --quick

# Custom datasets only
./train_all_classifiers.sh --datasets cifar10,cifar100

# Custom architectures
./train_all_classifiers.sh --archs resnet18,resnet50,vgg16

# Custom batch size and learning rate
./train_all_classifiers.sh --batch_size 256 --lr 0.02
```

**Options:**
- `--gpu N`: GPU device to use (default: 0)
- `--epochs N`: Number of epochs (default: 50)
- `--batch_size N`: Batch size (default: 128)
- `--lr FLOAT`: Learning rate (default: 0.01)
- `--quick`: Quick mode - only trains ResNet18 and ResNet50
- `--datasets LIST`: Comma-separated dataset list
- `--archs LIST`: Comma-separated architecture list

### `train_quick_test.sh`
Quick test script for validation before full training run.

**Usage:**
```bash
./train_quick_test.sh [GPU_ID]

# Examples:
./train_quick_test.sh     # Uses GPU 0
./train_quick_test.sh 1   # Uses GPU 1
```

## Training Configuration

### Default Settings
- **Pretrained weights**: ImageNet pretrained (enabled by default)
- **Image sizes**: 32×32 (CIFAR), 64×64 (TinyImageNet)
- **Normalization**: ImageNet statistics
- **Epochs**: 50
- **Batch size**: 128
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.01
- **Scheduler**: Cosine annealing
- **Mixed precision**: Enabled (on CUDA)

### Output

**Models saved to:**
```
./model_zoo/trained_model/<arch>_<dataset>.pth
```

**Logs saved to:**
```
./logs/training_<timestamp>/<arch>_<dataset>.log
```

## Expected Training Times

Approximate times on a single GPU (depends on GPU model):

| Architecture | CIFAR-10/100 (50 epochs) | TinyImageNet (50 epochs) |
|--------------|--------------------------|--------------------------|
| ResNet18     | ~20-30 min              | ~60-90 min               |
| ResNet50     | ~40-60 min              | ~120-180 min             |
| VGG16        | ~30-45 min              | ~90-150 min              |
| ViT-B/16     | ~60-90 min              | ~180-240 min             |

**Full training run** (all 8 architectures × 3 datasets = 24 models):
- Estimated time: 24-48 hours on a single GPU
- Use `--quick` mode to train only ResNet variants

## Monitoring Training

### During Training
The script shows:
- Progress bars with loss values
- Validation accuracy per epoch
- Best model checkpoint saves

### After Training
Check logs for detailed training history:
```bash
cat ./logs/training_*/resnet18_cifar100.log
```

List all trained models:
```bash
ls -lh ./model_zoo/trained_model/
```

## Tips

1. **Start with quick test**: Always run `train_quick_test.sh` first
2. **Use quick mode**: Try `--quick` mode before full training
3. **Monitor GPU**: Use `nvidia-smi` to monitor GPU usage
4. **Resume failed runs**: Re-run specific combinations that failed
5. **Disk space**: Ensure ~10GB free space for models and logs

## Advanced Usage

### Train Specific Combination
```bash
python fit_classifiers.py \
    --dataset cifar100 \
    --arch resnet18 \
    --pretrained true \
    --img_size 32 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.01 \
    --weight_decay 5e-4 \
    --label_smoothing 0.1
```

### Use Dataset-Specific Normalization
```bash
python fit_classifiers.py \
    --dataset cifar100 \
    --arch resnet18 \
    --pretrained true \
    --img_size 32 \
    --use_imnet_stats false  # Use CIFAR stats instead of ImageNet
```

### Train on Multiple GPUs
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 ./train_all_classifiers.sh --gpu 0 --datasets cifar10

# Terminal 2
CUDA_VISIBLE_DEVICES=1 ./train_all_classifiers.sh --gpu 0 --datasets cifar100
```

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` (try 64 or 32)
- Use smaller architecture (ResNet18 instead of ResNet50)

**Dataset not found:**
- CIFAR datasets will auto-download to `./dataset`
- TinyImageNet: Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip
  and extract to `./dataset/tiny-imagenet-200/`

**CUDA errors:**
- Verify GPU availability: `nvidia-smi`
- Try `--device cpu` for CPU training (much slower)
