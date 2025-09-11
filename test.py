# 假设 head 是你训练/加载好的 GMM（xdep=False）
from utils import viz_gmm

import os
from types import SimpleNamespace

import torch

# our utils
from utils import (
    get_dataset, build_model, eval_acc, parse_batch_spec,
    build_encoder, load_decoder_backend, g_ball
)
from fit_gmm import GMM  # our GMM class (including compute_pr_on_clean_correct)

# path to GMM package & classifier ckpt
PKG_PATH  = "./log/gmm_ckp/x_indep/gmm_CE_resnet18_cifar10_cov(diag)_L(linf_0p0314)_K(3).pt"
CLF_CKPT  = "./model_zoo/trained_model/sketch/resnet18_cifar10.pth"

# optional settings
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
S_SAMPLES = 100        # PR number of MC samples per image
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


# x-independent
from utils.viz_gmm import visualize_gmm_components_notebook
_ = visualize_gmm_components_notebook(
    head=gmm,
    as_correlation=True,
    max_cov_dim=32,
    S_latent=3000,
    gridsize=200,
    use_decoder=False,          # 若你的 GMM 是 pixel-space
    g_ball_fn=g_ball,           # 你项目里的投影函数
    gamma=8/255, norm_type="linf",
    save_dir="viz/gmm_components"  # 想保存就给目录；只在Notebook显示则设为 None
)

# x-dependent（举例）
# x_sample = next(iter(loader))[0][0]   # 取一张图 (C,H,W)
# _ = visualize_gmm_components_notebook(
#     head=gmm, encoder=encoder, x_for_xdep=x_sample,
#     out_shape=(C,H,W), g_ball_fn=g_ball, use_decoder=True,  # 若你的模型配了 decoder
#     as_correlation=True, max_cov_dim=32, S_latent=3000, gridsize=200,
#     gamma=8/255, norm_type="linf", save_dir="viz/gmm_components_xdep"
# )
