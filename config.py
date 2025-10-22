# config.py
"""
Configuration file for GMM4PR experiments.
All hyperparameters in one place.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Experiment configuration."""
    # Experiment name
    exp_name: str = "debug"

    # Dataset
    dataset: str = "cifar10"  # cifar10, cifar100, tinyimagenet
    data_root: str = "./data"
    
    # Model Architecture
    arch: str = "resnet18"
    clf_ckpt: str = "./model_zoo/trained_model/resnet18_cifar10.pth"
    
    # GMM Settings
    K: int = 3  # Number of mixture components
    latent_dim: int = 64
    cond_mode: str = "xy"  # x, y, xy, None
    cov_type: str = "diag"  # diag, lowrank, full
    cov_rank: int = 0  # For lowrank only
    hidden_dim: int = 256
    
    # Label Embedding
    use_y_embedding: bool = True
    y_emb_dim: int = 64
    y_emb_normalize: bool = True
    
    # Decoder
    use_decoder: bool = True  # If False the latent dimension should match input dimension
    decoder_backend: str = "bicubic"  # bicubic, conv, mlp, etc.
    
    # Perturbation Budget
    norm: str = "linf"  # linf, l2
    epsilon: float = 8/255
    
    # Training
    epochs: int = 5  # Small for debugging
    batch_size: int = 256  # Small for debugging
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    accumulate_grad: int = 1
    
    # Loss
    loss_variant: str = "cw"  # cw, ce
    kappa: float = 0.0
    num_samples: int = 10  # MC samples per image
    chunk_size: int = 10
    
    # Regularization
    reg_pi_entropy: float = 1
    # reg_comp_usage: float = 0.001
    reg_mean_div: float = 1
    # reg_kl_prior: float = 0.0001
    
    # Temperature Schedule (for distribution parameters)
    T_pi_init: float = 1.2
    T_pi_final: float = 1.0
    T_mu_init: float = 1.0
    T_mu_final: float = 1.0
    T_sigma_init: float = 1.5
    T_sigma_final: float = 1.0
    T_shared_init: float = 1.0
    T_shared_final: float = 1.0
    warmup_epochs: int = 3
    
    # Gumbel-Softmax Temperature (for reparameterized sampling)
    use_gumbel_anneal: bool = False  # Enable temperature annealing
    gumbel_temp_init: float = 1.0   # Initial temperature (softer)
    gumbel_temp_final: float = 0.5  # Final temperature (more discrete)
    
    # Initialization
    init_mode: str = "spread"  # spread, random, grid, uniform
    
    # Monitoring & Logging
    check_collapse_every: int = 2  # Check mode collapse every N epochs
    ckp_dir: str = "./ckp/gmm_ckp/"
    # save_freq: int = 5
    
    # Device
    device: str = "cuda"
    num_workers: int = 2  # Small for debugging
    seed: int = 42
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("=" * 60)
        for key, value in self.__dict__.items():
            lines.append(f"  {key:25s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============ PREDEFINED CONFIGS ============

def get_config(name: str = "debug") -> Config:
    """
    Get configuration by name.
    
    Args:
        name: Config name
    
    Returns:
        Config object
    """
    configs = {
        "debug": Config(
            exp_name="debug",
            K=3,
            epochs=5,
            batch_size=32,
            num_samples=8,
            check_collapse_every=2,
            warmup_epochs=3,
            use_gumbel_anneal=False,  # No annealing for debug
            gumbel_temp_init=1.0,
        ),
        
        # Example: Config with Gumbel annealing
        "debug_anneal": Config(
            exp_name="debug_anneal",
            K=3,
            epochs=10,
            batch_size=32,
            num_samples=8,
            check_collapse_every=2,
            warmup_epochs=5,
            use_gumbel_anneal=True,   # Enable annealing
            gumbel_temp_init=1.0,     # Start soft
            gumbel_temp_final=0.3,    # End more discrete
        ),
        
        # You can add more configs later:
        # "baseline": Config(...),
        # "k7_full": Config(...),
    }
    
    if name not in configs:
        raise ValueError(
            f"Unknown config: '{name}'\n"
            f"Available: {list(configs.keys())}"
        )
    
    return configs[name]


def list_configs():
    """List all available configs."""
    print("\nAvailable configurations:")
    print("=" * 60)
    print("  debug        - Minimal setup for debugging (K=3, 5 epochs, no Gumbel annealing)")
    print("  debug_anneal - Debug with Gumbel temperature annealing (K=3, 10 epochs)")
    print("=" * 60)