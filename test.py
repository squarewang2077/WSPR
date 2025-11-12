import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import os
import torch
from utils import GMM4PR, get_dataset, build_model, build_decoder_from_flag
from typing import Tuple, Dict, Any
# Define CSV files to analyze
# Get all CSV files from the directory using pathlib

# csv_dir = Path('./ckp/gmm_ckp/resnet18_on_cifar10')
# csv_dir = Path('./ckp/gmm_ckp/resnet50_on_cifar10')
csv_dir = Path('./ckp/gmm_ckp/resnet18_on_tinyimagenet')
# csv_dir = Path('./ckp/gmm_ckp/vgg16_on_cifar10')

all_csv_files = sorted([str(f) for f in csv_dir.glob('*.csv')])

print(f"Found {len(all_csv_files)} CSV files in {csv_dir}")

# Inspect first few files
def inspect_csv_columns(csv_files: List[str], last_row_idx: int = -1):
    """
    Inspect and display columns available in CSV files.
    
    Parameters
    ----------
    csv_files : List[str]
        List of CSV file paths to inspect
    """
    print("="*80)
    print("CSV File Column Inspection")
    print("="*80)
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"\nFile: {Path(csv_file).name}")
        print(f"  Shape: {df.shape} (rows, columns)")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Last few rows:")
        print(df.iloc[:last_row_idx].tail(3).to_string(index=False))
        print("-" * 80)


print("Function 'inspect_csv_columns' defined successfully!")


def load_gmm_config(gmm_path: str) -> Dict[str, Any]:
    """
    Load the configuration from a saved GMM checkpoint.
    
    Args:
        gmm_path: Path to the GMM checkpoint file (.pt)
        
    Returns:
        Dictionary containing both GMM config and full experiment config
    """
    if not os.path.isfile(gmm_path):
        raise FileNotFoundError(f"GMM checkpoint not found: {gmm_path}")
    
    ckpt = torch.load(gmm_path, map_location="cpu", weights_only=False)
    
    if "config" not in ckpt:
        raise ValueError(f"No config found in checkpoint: {gmm_path}")
    
    gmm_config = ckpt["config"]
    
    # The full experiment config is stored inside gmm_config["config"]
    exp_config = gmm_config.get("config", {})
    
    print(f"✓ Loaded config from: {gmm_path}")
    print(f"  Experiment: {exp_config.get('exp_name', 'N/A')}")
    print(f"  Dataset: {exp_config.get('dataset', 'N/A')}")
    print(f"  Architecture: {exp_config.get('arch', 'N/A')}")
    print(f"  K: {gmm_config['K']}, latent_dim: {gmm_config['latent_dim']}")
    print(f"  Condition mode: {gmm_config.get('cond_mode', 'None')}")
    print(f"  Covariance type: {gmm_config.get('cov_type', 'diag')}")
    print(f"  Perturbation: {gmm_config.get('budget', {})}")
    
    return gmm_config


def load_gmm_model(gmm_path: str, device: str = "cuda") -> Tuple[GMM4PR, Any, Any, Any]:
    """
    Load a complete GMM model from checkpoint with all necessary components.
    
    Args:
        gmm_path: Path to the GMM checkpoint file (.pt)
        device: Device to load the model to ('cuda' or 'cpu')
        
    Returns:
        Tuple of (gmm_model, classifier_model, feat_extractor, up_sampler)
    """
    # Step 1: Load configuration
    gmm_config = load_gmm_config(gmm_path)
    exp_config = gmm_config.get("config", {})
    
    # Extract necessary config values
    dataset_name = exp_config.get("dataset", "cifar10")
    data_root = exp_config.get("data_root", "./dataset")
    arch = exp_config.get("arch", "resnet18")
    clf_ckpt = exp_config.get("clf_ckpt")
    use_decoder = exp_config.get("use_decoder", False)
    decoder_backend = exp_config.get("decoder_backend", "bicubic")
    latent_dim = gmm_config["latent_dim"]
    
    # Step 2: Load dataset to get output shape
    print(f"\n{'='*60}")
    print("Loading dataset...")
    dataset, num_classes, out_shape = get_dataset(dataset_name, data_root, train=True, resize=False)
    print(f"  Dataset: {dataset_name}, Classes: {num_classes}, Shape: {out_shape}")
    
    # Step 3: Load classifier and feature extractor
    print(f"\n{'='*60}")
    print(f"Loading classifier: {arch}")
    print(f"  Device: {device}")
    
    if not clf_ckpt or not os.path.isfile(clf_ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {clf_ckpt}")
    
    model, feat_extractor = build_model(arch, num_classes, device)
    
    # Load classifier weights
    state = torch.load(clf_ckpt, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state.get("model_state", state))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    
    feat_extractor = feat_extractor.to(device).eval()
    for p in feat_extractor.parameters():
        p.requires_grad = False
    
    # Check parameter sharing
    model_params = {id(p) for p in model.parameters()}
    feat_params = {id(p) for p in feat_extractor.parameters()}
    shared = model_params & feat_params
    
    print(f"  Model params: {len(model_params)}, Feat extractor params: {len(feat_params)}")
    if shared:
        print(f"  Shared parameters: {len(shared)}")
    
    # Step 4: Build decoder/up_sampler if needed
    up_sampler = None
    if use_decoder:
        print(f"\n{'='*60}")
        print(f"Building decoder: {decoder_backend}")
        up_sampler = build_decoder_from_flag(
            decoder_backend,
            latent_dim,
            out_shape,
            device
        )
        print(f"  ✓ Decoder built successfully!")
    else:
        print(f"\n{'='*60}")
        print(f"No decoder used (use_decoder={use_decoder})")
    
    # Step 5: Load the GMM model
    print(f"\n{'='*60}")
    print(f"Loading GMM model from: {gmm_path}")
    gmm = GMM4PR.load_from_checkpoint(
        gmm_path,
        feat_extractor=feat_extractor,
        up_sampler=up_sampler,
        map_location=device,
        strict=True
    )
    
    gmm = gmm.to(device).eval()
    print(f"✓ GMM model loaded successfully!")
    print(f"{'='*60}\n")
    
    return gmm, gmm_config, model, feat_extractor, up_sampler


print("Functions defined: load_gmm_config(), load_gmm_model()")


keywords = ['loss_hist', 'cond(xy)', 'K7', 'decoder(trainable_128)', 'linf(16)' ,'reg(none)']
keywords_entropy = keywords.copy()
keywords_entropy[0] = 'collapse_log'
filtered_csv_files = [f for f in all_csv_files if all(k in f for k in keywords)]
filtered_csv_files_entropy = [f for f in all_csv_files if all(k in f for k in keywords_entropy)]

# Create corresponding GMM model paths from the filtered CSV files
gmm_model_paths = [f.replace('loss_hist_', 'gmm_').replace('.csv', '.pt') for f in filtered_csv_files]
print("\n".join([f"{n}" for n in filtered_csv_files]))
print("\n".join([f"{n}" for n in filtered_csv_files_entropy]))
print("\n".join([f"{n}" for n in gmm_model_paths]))

from utils import compute_pr_on_clean_correct
# Specify the GMM checkpoint path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "tinyimagenet"

res_collection = {}
# Load the complete GMM model
for GMM_PATH in gmm_model_paths:
    gmm, cfg, model, feat_extractor, up_sampler = load_gmm_model(GMM_PATH, device=DEVICE)

    test_dataset, num_classes, out_shape = get_dataset(DATASET, "./dataset", train=True, resize=False)
    # Create DataLoader for evaluation
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Set number of samples per image
    S_SAMPLES = 500  # Number of Monte Carlo samples per image

    # Optional: Specify batch indices to evaluate (None = all batches)
    # BATCH_INDICES = None  # Evaluate all batches
    # BATCH_INDICES = [0, 1, 2, 3, 4]  # Evaluate only first 5 batches
    BATCH_INDICES = range(1000) 

    print(f"\nComputing PR on clean-correct samples...")
    print(f"  GMM: {cfg['config']['exp_name']}")
    print(f"  Samples per image: {S_SAMPLES}")
    print(f"  Batch indices: {'All' if BATCH_INDICES is None else BATCH_INDICES}")

    # Compute PR
    pr, n_used, clean_acc = compute_pr_on_clean_correct(
        model=model,
        gmm=gmm,
        loader=loader,
        out_shape=out_shape,
        device=DEVICE,
        num_samples=S_SAMPLES,
        batch_indices=BATCH_INDICES,
        temperature=1,  # Very low temperature should approximate hard sampling
        use_soft_sampling=True,
        chunk_size=32
    )

    res_collection[GMM_PATH] = pr

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"  Probabilistic Robustness (PR): {pr}")
    print(f"  Clean-correct samples used: {n_used}")
    print(f"{'='*60}")

    print("\nAll components loaded and ready for analysis!")
