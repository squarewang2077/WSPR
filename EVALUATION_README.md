# Robustness Evaluation Tool

This tool (`test.py`) evaluates classifier robustness using:
1. **Baseline Noise**: Gaussian and Uniform distributions
2. **Adversarial Attacks**: PGD and CW attacks

All evaluations are performed on **clean-correct samples** (samples correctly classified by the model).

## Quick Start

```bash
python test.py \
    --dataset cifar10 \
    --arch resnet18 \
    --clf_ckpt ./ckp/resnet18_on_cifar10.pt \
    --epsilon 0.03137 \
    --num_samples 500 \
    --attack_steps 20
```

## Command Line Arguments

### Required Arguments

- `--dataset`: Dataset name (`cifar10`, `cifar100`, `tinyimagenet`)
- `--arch`: Model architecture (e.g., `resnet18`, `vgg16`, `wide_resnet50_2`)
- `--clf_ckpt`: Path to classifier checkpoint file

### Evaluation Parameters

- `--epsilon`: Perturbation budget (default: `8/255 = 0.03137`)
  - Use `0.03137` for 8/255
  - Use `0.06274` for 16/255 (to match GMM experiments)
- `--norm_type`: Norm constraint (`linf` or `l2`, default: `linf`)
- `--num_samples`: Number of samples for baseline noise (default: `500`)
- `--chunk_size`: Chunk size for memory management (default: `8`)

### Attack Parameters

- `--attack_steps`: Number of attack iterations (default: `20`)
- `--step_size`: Step size for attacks (default: `2/255 = 0.00784`)
- `--cw_margin`: CW attack margin/kappa (default: `2.0`)

### Data Loading

- `--batch_size`: Batch size (default: `128`)
- `--num_workers`: Number of workers (default: `2`)
- `--max_batches`: Limit evaluation to N batches (default: `None` = all)

### Output

- `--log_dir`: Directory for log files (default: `./logs`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

## Output Format

The script generates a timestamped log file in the format:
```
logs/eval_{dataset}_{arch}_{timestamp}.txt
```

### Example Log File Content

```
================================================================================
Robustness Evaluation Report
Generated: 2025-11-11 14:30:00
================================================================================

Dataset: cifar10
Architecture: resnet18
Checkpoint: ./ckp/resnet18_on_cifar10.pt
Device: cuda
Epsilon: 0.0314 (8.0/255)
Norm: linf
Num samples (baseline): 500
Attack steps: 20
Attack step size: 0.0078 (2.0/255)
CW margin: 2.0

================================================================================

GAUSSIAN BASELINE
--------------------------------------------------------------------------------
Clean Accuracy: 95.23%
Probabilistic Robustness (PR): 0.4521
Clean-correct samples: 9523

UNIFORM BASELINE
--------------------------------------------------------------------------------
Clean Accuracy: 95.23%
Probabilistic Robustness (PR): 0.4312
Clean-correct samples: 9523

PGD ATTACK
--------------------------------------------------------------------------------
Clean Accuracy: 95.23%
Robust Accuracy (on clean-correct): 42.15%
Attack Success Rate: 57.85%

CW ATTACK
--------------------------------------------------------------------------------
Clean Accuracy: 95.23%
Robust Accuracy (on clean-correct): 38.92%
Attack Success Rate: 61.08%

================================================================================
SUMMARY
================================================================================

Baseline Noise:
  Gaussian PR: 0.4521
  Uniform PR:  0.4312

Adversarial Attacks (on clean-correct):
  PGD Attack Success: 57.85%
  CW Attack Success:  61.08%

  PGD Robust Accuracy: 42.15%
  CW Robust Accuracy:  38.92%

================================================================================
```

## Usage Examples

### Example 1: Quick Test (100 batches)
```bash
python test.py \
    --dataset cifar10 \
    --arch resnet18 \
    --clf_ckpt ./ckp/resnet18_on_cifar10.pt \
    --epsilon 0.03137 \
    --max_batches 100
```

### Example 2: Full Evaluation with 16/255 epsilon (GMM setting)
```bash
python test.py \
    --dataset cifar10 \
    --arch resnet18 \
    --clf_ckpt ./ckp/resnet18_on_cifar10.pt \
    --epsilon 0.06274 \
    --num_samples 500 \
    --attack_steps 20
```

### Example 3: TinyImageNet Evaluation
```bash
python test.py \
    --dataset tinyimagenet \
    --arch vgg16 \
    --clf_ckpt ./ckp/vgg16_on_tinyimagenet.pt \
    --epsilon 0.03137 \
    --num_samples 500 \
    --batch_size 64
```

### Example 4: Using the Shell Script
```bash
# Edit run_evaluation.sh with your parameters
./run_evaluation.sh
```

## Key Metrics Explained

### Baseline Noise Metrics
- **Clean Accuracy**: Percentage of correctly classified samples
- **Probabilistic Robustness (PR)**: Average probability that perturbed samples remain correctly classified
- **Clean-correct samples**: Number of samples used for evaluation

### Attack Metrics
- **Clean Accuracy**: Percentage of correctly classified samples (should match baseline)
- **Robust Accuracy**: Percentage of clean-correct samples that remain correct after attack
- **Attack Success Rate**: Percentage of clean-correct samples successfully fooled by the attack
  - Formula: `100% - Robust Accuracy`
  - Higher = More effective attack

## Notes

1. **All evaluations use clean-correct samples**: Only samples that the model classifies correctly without perturbation are used for robustness evaluation.

2. **PR vs Attack Success**:
   - PR measures average success over random perturbations
   - Attack success measures worst-case adversarial perturbations

3. **Memory management**: Use `--chunk_size` to control memory usage for baseline evaluations with large `--num_samples`.

4. **Epsilon values**:
   - Standard adversarial: 8/255 = 0.03137
   - GMM experiments: 16/255 = 0.06274

5. **Progress bars**: The script shows progress bars during evaluation. Disable with `tqdm` configuration if running in batch mode.
