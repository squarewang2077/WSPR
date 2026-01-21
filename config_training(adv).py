# config.py
"""
Configuration file for adversarial training and finetuning.
All hyperparameters in one place.
"""
from dataclasses import dataclass

@dataclass
class BasicConfig:
    """Basic configuration."""
    # Device
    device: str = "cuda"
    num_workers: int = 2  # Small for debugging
    seed: int = 42

    # Initialization for unconditional GMM
    init_mode: str = "uniform"  # spread, random, grid, uniform
    
    # Monitoring & Logging
    check_collapse_every: int = 10  # Check mode collapse every N epochs
    ckp_dir: str = "./ckp/gmm_fitting"
    

@dataclass
class Config(BasicConfig):
    """Experiment configuration."""
    # Experiment name
    exp_name: str = "test_experiment"

    # Dataset
    dataset: str = "cifar10"  # cifar10, cifar100, tinyimagenet
    data_root: str = "./dataset"
    resize: bool = False  # Resize images to 64x64 for tinyimagenet 
    
    # Model Architecture
    arch: str = "resnet18"
    clf_ckpt: str = "./model_zoo/trained_model/sketch/resnet18_cifar10.pth"

    ### experiment-specific settings ###
    # GMM settings
    K: int = 7  # 1, 3, 7, 20
    latent_dim: int = 128  # For CIFAR10 without compression

    # Condition settings
    cond_mode: str | None = 'xy'  # x, y, xy, None
    cov_type: str = "full"  # diag, lowrank, full
    cov_rank: int = 0  # For lowrank only
    hidden_dim: int = 512

    # Label Embedding
    use_y_embedding: bool = True
    y_emb_dim: int = 128
    y_emb_normalize: bool = True

    # Decoder
    use_decoder: bool = True
    decoder_backend: str = "bicubic_trainable"  # bicubic, conv, mlp, etc.

    # Perturbation Budget
    norm: str = 'linf'
    epsilon: float = 16/255  # 4/255 8/255 16/255
    ### experiment-specific hyperparameters ###

    # Training Hyperparameters
    epochs: int = 50  
    batch_size: int = 128  
    batch_index_max: int = float("inf")  # For debugging, limit number of batches per epoch 
    
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    accumulate_grad: int = 1

    # Learning Rate Scheduler
    use_lr_scheduler: bool = False  # Enable Cosine Annealing with Warmup
    lr_warmup_epochs: int = 20  # Number of warmup epochs
    lr_min: float = 2e-6  # Minimum learning rate for cosine annealing
    
    # Loss
    loss_variant: str = "cw"  # cw, ce
    kappa: float = 1

    # Sampling
    num_samples: int = 32  # MC samples per image
    chunk_size: int = 32
    
    # Regularization
    reg_pi_entropy: float = 0.0
    reg_mean_div: float = 0.0 
    
    # Temperature Schedule 
    T_pi_init: float = 3.0
    T_pi_final: float = 1.0

    T_mu_init: float = 3.0
    T_mu_final: float = 1.0

    T_sigma_init: float = 1.5
    T_sigma_final: float = 1.0

    T_shared_init: float = 1.5
    T_shared_final: float = 1.0
    warmup_epochs: int = 50  # Number of warmup epochs for temperature annealing
    
    # Gumbel-Softmax Temperature (for reparameterized sampling)
    use_gumbel_anneal: bool = True  # Enable temperature annealing
    gumbel_temp_init: float = 1.0   # Initial temperature (softer)
    gumbel_temp_final: float = 0.1  # Final temperature (more discrete)
    
    
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

        "resnet18_on_cifar10_linf": Config(
            # Experiment name
            exp_name = "K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none)",

            # learning settings for none 
            lr = 5e-4,
            use_lr_scheduler = False,
            # lr_min = 2e-6,  # Minimum learning rate for cosine annealing
            # lr_warmup_epochs = 5,  # Number of warmup epochs
            warmup_epochs = 10,  # Number of warmup epochs for temperature annealing

            # GMM settings
            K = 7,  # 3, 7, 12
            latent_dim = 128,  

            # Condition settings
            cond_mode = "xy",  # x, y, xy, None
            cov_type = "full",  # diag, lowrank, full
            cov_rank = 0,  # For lowrank only
            hidden_dim = 256,

            # Label Embedding
            use_y_embedding = True,
            y_emb_dim = 64,
            y_emb_normalize = True,

            # Decoder
            use_decoder = True,
            decoder_backend = 'bicubic_trainable',  # bicubic, conv, mlp, etc.

            # Perturbation Budget
            norm = "linf",
            epsilon = 16/255, # 4/255 8/255 16/255

        ),

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
    print("  resnet18_on_cifar10(cuda0) - K=1, unconditional, no decoder, linf epsilon=4/255")
    print("=" * 60)