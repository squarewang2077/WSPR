# Notebook-friendly imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from types import SimpleNamespace

import torch

# our utils
from utils import (
    get_dataset, build_model, eval_acc, parse_batch_spec,
    build_encoder, load_decoder_backend, g_ball
)
from fit_gmm import GMM  # our GMM class (including compute_pr_on_clean_correct)

# path to GMM package & classifier ckpt
PKG_PATH  = "./log/gmm_ckp/x_dep/gmm_CWlike_resnet18_cifar10_cov(lowrank)_L(l2_0p2500)_K(7).pt"
CLF_CKPT  = "./model_zoo/trained_model/sketch/resnet18_cifar10.pth"

# optional settings
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
S_SAMPLES = 10        # PR number of MC samples per image
BATCH_IDX = ""         # the batch index，e.g. "0,3,5-10"；"" means all batches
BATCH_SIZE= 128        # the batch size for the evaluation DataLoader

device = torch.device(DEVICE)

if not os.path.isfile(PKG_PATH):
    raise FileNotFoundError(PKG_PATH)

# let out_shape=None first，refill out_shape later after we load dataset
# (because out_shape depends on dataset)
gmm, enc_loaded, dec_loaded, meta = GMM.load_package(
    filepath=PKG_PATH,
    device=device,
    build_encoder_fn=build_encoder,
    load_decoder_backend_fn=load_decoder_backend,
    out_shape=None
)

# use snapshot to rebuild dataset, model, feat_extractor
snap = meta.get("args_snapshot") or {}
eva  = SimpleNamespace(**snap)  # eva.dataset / eva.arch / eva.gamma / eva.norm / ...

# addtional settings (override snapshot) for classfier ckpt & eval
clf_ckpt = getattr(eva, "clf_ckpt", None) or CLF_CKPT
print("Loaded package:", PKG_PATH)
print("Snapshot keys:", list(snap.keys()))


dataset, num_classes, out_shape = get_dataset(eva.dataset, train=False)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
print("Dataset:", eva.dataset, "num_classes:", num_classes, "out_shape:", out_shape)


model, feat_extractor = build_model(eva.arch, num_classes, device)

# load classifier ckpt
state = torch.load(clf_ckpt, map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[clf] loaded. missing={len(missing)} unexpected={len(unexpected)}")

# freeze & eval()
model = model.to(device).eval();  [p.requires_grad_(False) for p in model.parameters()]
feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()]

# check accuracy on clean data
acc_clean = eval_acc(model, dataset, device)
print(f"[clean acc] {acc_clean*100:.2f}%")


from utils import compute_pr_on_clean_correct_old
# whether use encoder & decoder and xdep
# all from the loaded GMM package
xdep        = meta["gmm_config"]["xdep"] 
use_decoder = meta["decoder_info"].get("use_decoder", False)
encoder_for_eval = enc_loaded if xdep else None
decoder_for_eval = dec_loaded if use_decoder else None

# the batch to be evaluated
pr_sel = parse_batch_spec(BATCH_IDX)

# compute PR (on clean correct samples)


pr, n_used, clean_acc = gmm.compute_pr_on_clean_correct(
    model=model,
    loader=loader,
    out_shape=out_shape,
    encoder=encoder_for_eval,
    decoder=decoder_for_eval,
    S=S_SAMPLES,
    gamma=eva.gamma,          # from training snapshot
    norm_type=eva.norm,       # from training snapshot
    use_decoder=use_decoder,
    batch_indices=pr_sel
)
print(f"[PR] used={n_used}  clean-acc={clean_acc*100:.2f}%  PR={pr:.4f}")
