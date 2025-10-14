import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily, MultivariateNormal, LowRankMultivariateNormal
from utils import *
from tqdm import tqdm

class GMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def _build_dist(self):  
        pass

    def sample(self):
        pass

    def _pr_loss(self):
        pass

    def fit(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


def _slug_gamma(g):
    # make gamma filename-safe, e.g. 0.03137255 -> 0p0314
    return f"{g:.4f}".replace('.', 'p')





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100","tinyimagenet"], default="cifar10")
    ap.add_argument("--arch", choices=["resnet18"], default="resnet18")
    ap.add_argument("--clf_ckpt", type=str, default="./model_zoo/trained_model/resnet18_cifar10.pth", \
                    help="path to trained classifier checkpoint (required)")
    ap.add_argument("--device", default="cuda")
 

    # GMM settings
    ap.add_argument("--num_modes", type=int, default=7) # plan to try 1,3,7,12,20 but for small model
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default="full") # we will based on full matrix here
    ap.add_argument("--cov_rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200) # 3 epochs for debug
    ap.add_argument("--lr", type=float, default=5e-4) # 2e-2 when x independent, 5e-4 when x-dependent
    ap.add_argument("--gamma", type=float, default=8/255) # 8/255 for Linf, 0.5 for L2
    ap.add_argument("--mc", type=int, default=10, \
                    help="MC samples per image per step")
    ap.add_argument("--xdep", default=True, action="store_true") # store True for test
    ap.add_argument("--norm", choices=["l2","linf"], default="linf")
    ap.add_argument("--batch_size", type=int, default=8) 



    args = ap.parse_args()
    cfg_str = f"CWlike_{args.arch}_{args.dataset}_cov({args.cov_type})_L({args.norm}_{_slug_gamma(args.gamma)})_K({args.num_modes})_Dec({args.decoder_backend if args.use_decoder else 'pixel'})"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset, num_classes, out_shape = get_dataset(args.dataset, train=False, resize=False) # always resize to 224 for imagenet pretrained encoders
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, feat_extractor = build_model(args.arch, num_classes, device)

    # Load classifier ckpt
    if not args.clf_ckpt or not os.path.isfile(args.clf_ckpt):
        raise ValueError("You must provide --clf_ckpt pointing to a trained classifier on this dataset.")
    
    state = torch.load(args.clf_ckpt, map_location="cpu") # map_location for debug on my laptop
    if "state_dict" in state: # this is for models that training from scratch
        # only leave the state dict in {"epoch": 10, "state_dict": model.state_dict(), "optimizer": opt.state_dict()}
        state = state["state_dict"]     
    elif "model_state" in state: # pretrained models after finetuning
        state = state["model_state"]
    state = {k.replace("module.",""): v for k,v in state.items()} # in case of dataparallel, trained with multi-gpu
    missing, unexpected = model.load_state_dict(state, strict=False) # check loading 
    print(f"[clf] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    # The downstream classifier and feat_extractor are frozen
    model = model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()] 
    feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()] # for safety


    # Collect all parameter IDs for both
    model_params = {id(p) for p in model.parameters()}
    feat_params  = {id(p) for p in feat_extractor.parameters()}

    # Check intersection
    shared = model_params & feat_params

    print(f"[check] model params: {len(model_params)}, feat_extractor params: {len(feat_params)}")
    if shared:
        print(f"[check] They share {len(shared)} parameters.")
    else:
        print("[check] No shared parameters.")


    # Check clean accuracy
    model.eval()
    eval_acc(model, dataset, device) 
        
    # Build GMM 

    # Save the fitted GMM 
    if args.use_decoder:
        save_root = os.path.join("./log/gmm_ckp", "x_dep/dec" if args.xdep else "x_indep/dec")
    else:
        save_root = os.path.join("./log/gmm_ckp", "x_dep" if args.xdep else "x_indep")
    os.makedirs(save_root, exist_ok=True)
    fname = f"gmm_{cfg_str}.pt"
    save_path = os.path.join(save_root, fname)



if __name__=="__main__":
    main()
