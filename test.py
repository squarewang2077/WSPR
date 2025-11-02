import os
import torch
from config import get_config
from utils import GMM4PR, get_dataset, build_model, build_decoder_from_flag, eval_acc, check_mode_collapse

# Specify which trained GMM to load
CONFIG_NAME = "test_1"  # Must match the config name used in fit_gmm2.py
CHECKPOINT_DIR = "./ckp/gmm_ckp/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load configuration
cfg = get_config(CONFIG_NAME)
checkpoint_path = os.path.join(CHECKPOINT_DIR, f"gmm_{cfg.exp_name}.pt")

# Load dataset to get out_shape
dataset, num_classes, out_shape = get_dataset(cfg.dataset, cfg.data_root, train=True, resize=True)
# Create DataLoader for evaluation
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)
print(f"Dataset: {cfg.dataset}, Classes: {num_classes}, Shape: {out_shape}")


# load classifier weights 
print(f"Using device: {DEVICE}")
print(f"\nLoading classifier: {cfg.arch}")
model, feat_extractor = build_model(cfg.arch, num_classes, DEVICE)

if not os.path.isfile(cfg.clf_ckpt):
    raise FileNotFoundError(f"Classifier not found: {cfg.clf_ckpt}")

state = torch.load(cfg.clf_ckpt, map_location="cpu")
state = state.get("state_dict", state.get("model_state", state))
state = {k.replace("module.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=False)
model = model.to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False

feat_extractor = feat_extractor.to(DEVICE).eval()
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
# eval_acc(model, dataset, DEVICE) # accuracy on training set

# Build decoder/up_sampler (if used during training)
up_sampler = None
if cfg.use_decoder:
    print(f"\nBuilding decoder: {cfg.decoder_backend}")
    up_sampler = build_decoder_from_flag(
        cfg.decoder_backend,
        cfg.latent_dim,
        out_shape,
        DEVICE
    )
    print(f"✓ Decoder {cfg.decoder_backend} built successfully!")
else:
    print(f"\nNo decoder used (use_decoder={cfg.use_decoder})")

# Load the trained GMM model
print(f"Loading from: {checkpoint_path}")
gmm = GMM4PR.load_from_checkpoint(
    checkpoint_path,
    feat_extractor=feat_extractor, 
    up_sampler=up_sampler,      
    map_location=DEVICE,
    strict=True
)

gmm = gmm.to(DEVICE).eval()


from utils import compute_pr_on_clean_correct


# Set number of samples per image
S_SAMPLES = 10  # Number of Monte Carlo samples per image

# Optional: Specify batch indices to evaluate (None = all batches)
# BATCH_INDICES = None  # Evaluate all batches
# BATCH_INDICES = [0, 1, 2, 3, 4]  # Evaluate only first 5 batches
BATCH_INDICES = range(10)  # Evaluate first 10 batches

print(f"\nComputing PR on clean-correct samples...")
print(f"  GMM: {cfg.exp_name}")
print(f"  Samples per image: {S_SAMPLES}")
print(f"  Batch indices: {'All' if BATCH_INDICES is None else BATCH_INDICES}")

# Compute PR
pr_soft, n_used, clean_acc = compute_pr_on_clean_correct(
    model=model,
    gmm=gmm,
    loader=loader,
    out_shape=out_shape,
    device=DEVICE,
    num_samples=100,
    batch_indices=range(10),
    temperature=0.001,  # Very low temperature should approximate hard sampling
    use_soft_sampling=True,
    chunk_size=32
)



check_mode_collapse(gmm, loader, DEVICE)

print(f"\n{'='*60}")
print(f"Results:")
print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
print(f"  Probabilistic Robustness (PR): {pr:.4f}")
print(f"  Clean-correct samples used: {n_used}")
print(f"{'='*60}")