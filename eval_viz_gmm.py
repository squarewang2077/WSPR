# eval_gmm.py
import os
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

# --- your project utils (dataset/model factories, helpers) ---
# must provide: get_dataset, build_model, eval_acc, parse_batch_spec,
#               build_encoder, load_decoder_backend, g_ball
from utils import *

# --- training / eval helpers you already have ---
# must provide: compute_pr_on_clean_correct
from fit_gmm import GMM


def _load_classifier(args, num_classes, device):
    """
    Build downstream classifier + feat_extractor and load trained weights.
    Both are set to eval() and frozen.
    """
    model, feat_extractor = build_model(args.arch, num_classes, device)

    # freeze model & feature extractor
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feat_extractor.eval()
    for p in feat_extractor.parameters():
        p.requires_grad_(False)

    # load classifier checkpoint
    if not args.clf_ckpt or not os.path.isfile(args.clf_ckpt):
        raise ValueError("You must provide --clf_ckpt pointing to a trained classifier on this dataset.")
    state = torch.load(args.clf_ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    # in case of DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[clf] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    return model, feat_extractor


def _make_eval_args(snapshot: dict, cli: argparse.Namespace) -> SimpleNamespace:
    """
    Build a minimal args namespace used by compute_pr_on_clean_correct & viz.
    Priority: CLI overrides > snapshot > sensible defaults.
    """
    snap = snapshot or {}

    def pick(key, default=None):
        return getattr(cli, key) if getattr(cli, key) is not None else snap.get(key, default)

    # carry over keys that downstream code expects to read
    ns = SimpleNamespace(
        dataset              = pick("dataset", "cifar10"),
        arch                 = pick("arch", "resnet18"),
        norm                 = pick("norm", "linf"),
        gamma                = pick("gamma", 8/255),
        mc                   = pick("mc", 1),
        num_modes            = snap.get("num_modes", 1),     # not on CLI
        cov_type             = pick("cov_type", snap.get("cov_type", "diag")),
        cov_rank             = snap.get("cov_rank", 0),      # not on CLI (only used for 'lowrank')
        xdep                 = snap.get("xdep", False),      # the saved model defines this
        use_decoder          = snap.get("use_decoder", False),
        encoder_backend      = snap.get("encoder_backend", "classifier"),
        decoder_backend      = snap.get("decoder_backend", "conv"),
        freeze_encoder       = snap.get("freeze_encoder", True),
        freeze_decoder       = snap.get("freeze_decoder", True),
        latent_dim           = snap.get("latent_dim", None),
        gan_class            = snap.get("gan_class", 207),
        gan_truncation       = snap.get("gan_truncation", 0.5),
    )
    return ns


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved GMM package: PR and (optional) contour viz.")
    # --- package & runtime ---
    ap.add_argument("--pkg", type=str, default="checkpoints/gmm_pkg.pt", help="Path to saved GMM package")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--S", type=int, default=100, help="MC samples for PR at eval time")
    ap.add_argument("--batch_idx", type=str, default="", help="Which batches to evaluate PR on. Empty=all.")
    # --- dataset / classifier for evaluation ---
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default=None,
                    help="(Optional override) dataset used for evaluation")
    ap.add_argument("--arch", choices=["resnet18","resnet50","wide_resnet50_2","vgg16",
                                       "densenet121","mobilenet_v3_large","efficientnet_b0","vit_b_16"],
                    default=None, help="(Optional override) downstream classifier arch")
    ap.add_argument("--clf_ckpt", type=str, default="./model_zoo/trained_model/ResNets/resnet18_cifar10.pth",
                    help="Path to trained classifier checkpoint")
    # --- overrides of a few robust params at eval ---
    ap.add_argument("--gamma", type=float, default=None, help="Override epsilon radius")
    ap.add_argument("--norm", choices=["l2", "linf"], default=None, help="Override norm type")
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default=None, help="Override cov type if needed")
    ap.add_argument("--mc", type=int, default=None, help="Override MC per step (used in some funcs)")
    # --- viz ---
    ap.add_argument("--viz", action="store_true", help="Draw PCA KDE contours")
    ap.add_argument("--n_inputs_when_xdep", type=int, default=8, help="How many inputs for x-dependent contours")
    ap.add_argument("--gridsize", type=int, default=200, help="KDE grid size")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- dataset & loader ---
    dataset, num_classes, out_shape = get_dataset(args.dataset or "cifar10", train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # --- classifier & feat_extractor (frozen) ---
    model, feat_extractor = _load_classifier(
        argparse.Namespace(arch=args.arch or "resnet18", clf_ckpt=args.clf_ckpt),
        num_classes=num_classes,
        device=device
    )

    # quick clean accuracy (helps sanity check)
    eval_acc(model, dataset, device)

    # --- load the saved GMM package and rebuild encoder/decoder ---
    gmm, enc_loaded, dec_loaded, meta = GMM.load_package(
        filepath=args.pkg,
        device=device,
        build_encoder_fn=build_encoder,
        load_decoder_backend_fn=load_decoder_backend,
        out_shape=out_shape
    )

    # --- build eval args (snapshot + CLI overrides) ---
    eval_args = _make_eval_args(meta.get("args_snapshot"), args)

    # --- decide encoder/decoder to use at inference ---
    xdep = meta["gmm_config"]["xdep"]
    encoder_for_eval = enc_loaded if xdep else None
    if (not xdep) and (eval_args.encoder_backend == "classifier"):
        # x-independent doesn't use an encoder, but some helper functions expect a value;
        # we'll pass None and the helper will handle it when xdep=False.
        pass
    decoder_for_eval = dec_loaded if meta["decoder_info"].get("use_decoder", False) else None

    # --- batch selection for PR eval ---
    pr_sel = parse_batch_spec(args.batch_idx)

    # --- print a short summary of what was loaded ---
    print("\n=== Loaded GMM package summary ===")
    print(f"  K={meta['gmm_config']['K']}, D={meta['gmm_config']['D']}, xdep={xdep}, cov_type={meta['gmm_config']['cov_type']}")
    print(f"  encoder: used={meta['encoder_info'].get('used', False)}, backend={meta['encoder_info'].get('backend')}, "
          f"trainable={meta['encoder_info'].get('trainable')}")
    print(f"  decoder: used={meta['decoder_info'].get('use_decoder', False)}, backend={meta['decoder_info'].get('backend')}, "
          f"trainable={meta['decoder_info'].get('trainable')}")
    print("  args snapshot:", meta.get("args_snapshot"))
    print("==================================\n")

    # --- evaluate PR on clean-correct using the loaded GMM ---
    pr, n_used, clean_acc = GMM.compute_pr_on_clean_correct(
        model=model,
        head=gmm,  # use loaded GMM
        encoder=(encoder_for_eval if xdep else (feat_extractor if eval_args.encoder_backend == "classifier" else None)),
        loader=loader,
        args=eval_args,
        out_shape=out_shape,
        decoder=decoder_for_eval,
        S=args.S,
        batch_indices=pr_sel
    )
    print(f"[PR] used={n_used} samples (clean acc={clean_acc*100:.2f}%), PR={pr:.4f}")

    # --- optional: draw PCA KDE contour visualizations ---
    if args.viz:
        # we try utils.viz_pca_contours first; if not found, try viz_contours module
        viz_fn = None
        try:
            from utils import viz_pca_contours as _viz_fn
            viz_fn = _viz_fn
        except Exception:
            try:
                from utils import viz_pca_contours as _viz_fn
                viz_fn = _viz_fn
            except Exception:
                viz_fn = None

        if viz_fn is None:
            print("[viz] No viz_pca_contours() found. Please place it in utils.py or viz_contours.py.")
        else:
            try:
                lat_png, eps_png = viz_fn(
                    loader=loader,
                    head=gmm,
                    encoder=(encoder_for_eval if xdep else (feat_extractor if eval_args.encoder_backend == "classifier" else None)),
                    decoder=decoder_for_eval,
                    build_gmm_fn=gmm.mixture,      # head.mixture(pi, mu, cov, cov_type)
                    g_ball_fn=g_ball,
                    args=eval_args,
                    out_shape=out_shape,
                    save_dir="viz",
                    n_inputs_when_xdep=args.n_inputs_when_xdep,
                    S_latent=5000,
                    gridsize=args.gridsize
                )
                print("[viz] contour images saved:", lat_png, eps_png)
            except Exception as e:
                print("[viz] failed to draw contours:", e)


if __name__ == "__main__":
    main()
