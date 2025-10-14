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
PKG_PATH  = "./log/gmm_ckp/x_dep/dec/gmm_CWlike_resnet18_cifar10_cov(full)_L(l2_0p5000)_K(3)_Dec(ae).pt"
CLF_CKPT  = "./model_zoo/trained_model/sketch/resnet18_cifar10.pth"

# optional settings
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
S_SAMPLES = 10        # PR number of MC samples per image
BATCH_IDX = "0-20"         # the batch index，e.g. "0,3,5-10"；"" means all batches
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
