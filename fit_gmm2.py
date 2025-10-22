"""
GMM4PR: Gaussian Mixture Model for Probabilistic Robustness
Simplified training script.

Usage:
    # Use default debug config
    python train_gmm.py
    
    # Override specific parameters
    python train_gmm.py --config debug --epochs 10 --K 7
    
    # List available configs
    python train_gmm.py --list-configs
"""

import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

# Import your classes and utilities
from config import get_config, list_configs
from utils import (
    GMM4PR,
    get_dataset, build_model, eval_acc,
    initialize_gmm_parameters, 
    TemperatureScheduler, check_mode_collapse,
    build_decoder_from_flag
)


def main():
    # ============ PARSE ARGUMENTS ============
    parser = argparse.ArgumentParser(description="GMM4PR Training")
    
    # Config selection
    parser.add_argument("--config", type=str, default="debug",
                       help="Config name from config.py")
    parser.add_argument("--list-configs", action="store_true", default=False, # false for debug
                       help="List all available configs and exit")
    
    # Quick overrides (optional)
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--K", type=int, help="Override number of components")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--clf_ckpt", type=str, help="Override classifier checkpoint path")
    
    args = parser.parse_args()
    
    # List configs and exit
    if args.list_configs:
        list_configs()
        return
    
    # Load config
    cfg = get_config(args.config)
    
    # Apply command line overrides
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.K is not None:
        cfg.K = args.K
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.clf_ckpt is not None:
        cfg.clf_ckpt = args.clf_ckpt
    
    # Print configuration
    print(cfg)
    
    # ============ SETUP ============
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Load dataset
    print(f"\nLoading dataset: {cfg.dataset}")
    dataset, num_classes, out_shape = get_dataset(cfg.dataset, train=True, resize=True) # training set
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    print(f"Dataset: {len(dataset)} samples, {num_classes} classes, shape={out_shape}")
    
    # ============ LOAD CLASSIFIER ============
    print(f"\nLoading classifier: {cfg.arch}")
    model, feat_extractor = build_model(cfg.arch, num_classes, device)
    
    if not os.path.isfile(cfg.clf_ckpt):
        raise FileNotFoundError(f"Classifier not found: {cfg.clf_ckpt}")
    
    state = torch.load(cfg.clf_ckpt, map_location="cpu")
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
    feat_params  = {id(p) for p in feat_extractor.parameters()}
    shared = model_params & feat_params

    print(f"[check] model params: {len(model_params)}, feat_extractor params: {len(feat_params)}")
    if shared:
        print(f"[check] They share {len(shared)} parameters.")
    else:
        print("[check] No shared parameters.")

    print("Evaluating clean accuracy...")
    eval_acc(model, dataset, device) # accuracy on training set
    
    # ============ INITIALIZE GMM ============
    print(f"\nInitializing GMM: K={cfg.K}, D={cfg.latent_dim}, cond={cfg.cond_mode}")
    gmm = GMM4PR(
        K=cfg.K,
        latent_dim=cfg.latent_dim,
        device=device,
        T_pi=cfg.T_pi_init,
        T_mu=cfg.T_mu_init,
        T_sigma=cfg.T_sigma_init,
        T_shared=cfg.T_shared_init
    )
    
    # Set label embedding
    if cfg.use_y_embedding:
        gmm.set_y_embedding(
            num_cls=num_classes,
            y_dim=cfg.y_emb_dim,
            normalize=cfg.y_emb_normalize
        )
    
    # Set regularization
    gmm.set_regularization(
        pi_entropy=cfg.reg_pi_entropy,
        # component_usage=cfg.reg_comp_usage,
        mean_diversity=cfg.reg_mean_div,
        # kl_prior=cfg.reg_kl_prior
    )
    
    # Infer feature dimension if needed
    feat_dim = None
    if cfg.cond_mode in ("x", "xy"):
        with torch.no_grad():
            x0, _, _ = next(iter(loader))
            feat_dim = feat_extractor(x0.to(device)).view(x0.size(0), -1).size(1)
        print(f"Feature dimension: {feat_dim}")
    
    # Set conditioning
    gmm.set_condition(
        cond_mode=cfg.cond_mode,
        cov_type=cfg.cov_type,
        cov_rank=cfg.cov_rank,
        feat_dim=feat_dim or 0,
        num_cls=num_classes,
        hidden_dim=cfg.hidden_dim
    )
    
    # Set feature extractor
    if cfg.cond_mode in ("x", "xy"):
        gmm.set_feat_extractor(feat_extractor)
    
    # Set decoder
    if cfg.use_decoder:
        decoder = build_decoder_from_flag(
            cfg.decoder_backend,
            cfg.latent_dim,
            out_shape,
            device
        )
        gmm.set_up_sampler(decoder)
    
    # Set budget
    gmm.set_budget(norm=cfg.norm, eps=cfg.epsilon)
    
    # Initialize parameters
    initialize_gmm_parameters(gmm, init_mode=cfg.init_mode) # only for unconditional GMM
    
    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        gmm,
        initial_T_pi=cfg.T_pi_init,
        final_T_pi=cfg.T_pi_final,
        initial_T_sigma=cfg.T_sigma_init,
        final_T_sigma=cfg.T_sigma_final,
        initial_T_shared=cfg.T_shared_init,
        final_T_shared=cfg.T_shared_final,
        warmup_epochs=cfg.warmup_epochs
    )
    
    # ============ OPTIMIZER ============
    optimizer = optim.Adam(
        [p for p in gmm.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # ============ TRAINING LOOP ============
    os.makedirs(cfg.ckp_dir, exist_ok=True)
    collapse_log = []
    loss_hist = {"epoch": [], 
                 "loss": [],
                 "main_loss": [],
                 "reg_loss": []
                 } # To store loss history

    gmm.train()
    print(f"\n{'='*60}")
    print(f"Starting training: {cfg.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.epochs + 1):
        # Update temperatures for distribution parameters
        T_pi, T_sigma, T_shared = temp_scheduler.step(epoch)
        
        # Compute Gumbel temperature (optional annealing)
        if hasattr(cfg, 'use_gumbel_anneal') and cfg.use_gumbel_anneal:
            alpha = epoch / cfg.epochs
            gumbel_temp = cfg.gumbel_temp_init + alpha * (cfg.gumbel_temp_final - cfg.gumbel_temp_init)
        else:
            gumbel_temp = 1.0  # Fixed temperature
        
        # Progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs} [T_π={T_pi:.2f}, T_g={gumbel_temp:.2f}]")
        
        # Metrics
        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_reg = 0.0
        total_samples = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y, _) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Only use correctly classified samples
            with torch.no_grad():
                pred = model(x).argmax(1)
                mask = (pred == y)
            
            if mask.sum() == 0:
                continue
            
            x_clean, y_clean = x[mask], y[mask]
            total_samples += len(y_clean)
            
            # Compute loss
            return_details = (batch_idx == 0 and epoch % cfg.check_collapse_every == 0)
            
            out = gmm.pr_loss(
                x_clean, y_clean, model,
                num_samples=cfg.num_samples,
                chunk_size=cfg.chunk_size,
                gumbel_temperature=gumbel_temp,  # Pass Gumbel temperature
                return_reg_details=return_details
            )
            
            loss = out["loss"] / cfg.accumulate_grad
            loss.backward()
            
            # Gradient step
            if (batch_idx + 1) % cfg.accumulate_grad == 0:
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(gmm.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Accumulate metrics
            epoch_loss += out["loss"].item()
            epoch_main += out["main"].item()
            epoch_reg += out["reg"].item()
            
            # Print regularization details
            if return_details and 'reg_details' in out:
                print(f"\n[Epoch {epoch}] Regularization details:")
                for k, v in out['reg_details'].items():
                    print(f"  {k:20s}: {v:.6f}")
                print(f"  π distribution: {out['pi_probs'].cpu().numpy()}")
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{out['loss'].item():.4f}",
                "main": f"{out['main'].item():.4f}",
                "reg": f"{out['reg'].item():.4f}",
            })
        
        # Final gradient step
        if len(loader) % cfg.accumulate_grad != 0:
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(gmm.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Epoch summary
        avg_loss = epoch_loss / max(len(loader), 1)
        avg_main = epoch_main / max(len(loader), 1)
        avg_reg = epoch_reg / max(len(loader), 1)
        
        # Record loss history
        loss_hist["epoch"].append(epoch)
        loss_hist["loss"].append(avg_loss)
        loss_hist["main_loss"].append(avg_main)
        loss_hist["reg_loss"].append(avg_reg)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {avg_loss:.4f} (main={avg_main:.4f}, reg={avg_reg:.4f})")
        print(f"  Samples used: {total_samples}/{len(dataset)}")
        print(f"  Temperatures: T_π={T_pi:.2f}, T_σ={T_sigma:.2f}, T_gumbel={gumbel_temp:.2f}")
        
        # Check mode collapse
        if epoch % cfg.check_collapse_every == 0:
            stats = check_mode_collapse(gmm, loader, device)
            collapse_log.append({
                'epoch': epoch,
                'max_pi': stats['max_pi'],
                'min_pi': stats['min_pi'],
                'std_pi': stats['std_pi'],
                'entropy_ratio': stats['entropy_ratio'],
                'T_pi': T_pi,
                'T_gumbel': gumbel_temp,
                'avg_loss': avg_loss
            })
            
            # Adaptive regularization
            if stats['max_pi'] > 0.6 and epoch < cfg.epochs * 0.8:
                print("⚠️  Mode collapse detected! Increasing regularization...")
                gmm.set_regularization(
                    pi_entropy=gmm.reg_coeffs['pi_entropy'] * 1.5,
                    component_usage=gmm.reg_coeffs['component_usage'] * 1.5
                )
    
    # ============ SAVE ============
    print(f"\n{'='*60}")
    print("Training complete! Saving model...")
    print(f"{'='*60}")
    
    # saving directory
    save_dir = cfg.ckp_dir
    os.makedirs(save_dir, exist_ok=True)

    # save the training loss
    pd.DataFrame(loss_hist).to_csv(os.path.join(save_dir, f"loss_hist_{cfg.exp_name}.csv"), index=False)
    print(f"[save] loss history -> {save_dir}/loss_hist_{cfg.exp_name}.csv")

    # Save model with metadata (including final gumbel_temp and configuration for reference)
    save_path = os.path.join(save_dir, f"gmm_{cfg.exp_name}.pt")
    gmm.save(
        save_path, 
        extra={
            "config": cfg.to_dict(),
            "final_gumbel_temperature": gumbel_temp,  # Final value used
        }
    )
    print(f"✓ Model saved: {save_path}")
    
    # Save collapse log
    if collapse_log:
        df = pd.DataFrame(collapse_log)
        log_path = os.path.join(save_dir, f"collapse_log_{cfg.exp_name}.csv")
        df.to_csv(log_path, index=False)
        print(f"✓ Collapse log saved: {log_path}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()