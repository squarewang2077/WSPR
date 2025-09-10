import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily, MultivariateNormal
from utils import *

# ------------------ GMM Head (diag/full/lowrank) --------------------
class GMM(nn.Module):
    """
    diag:     -> (pi, mu, sigma)            sigma: (B,K,D)
    full:     -> (pi, mu, L)                L:     (B,K,D,D)  (scale_tril)
    lowrank:  -> (pi, mu, (U, sigma_diag))  U:     (B,K,D,r), sigma_diag:(B,K,D)
    """
    def __init__(self, feat_dim, K, latent_dim, xdep, cov_type, cov_rank):
        super().__init__()
        self.K, self.D, self.xdep = K, latent_dim, xdep
        self.cov_type, self.r = cov_type, cov_rank
        hid = 512
        if xdep: # if the omega is x-dependent
            # here trunk is a simple 2-layer MLP, it can be removed or replaced by a more complex network
            self.trunk = nn.Sequential(nn.Linear(feat_dim, hid), nn.ReLU()) # feature to hidden
            self.pi_head = nn.Linear(hid, K) # hidden to pi logits
            self.mu_head = nn.Linear(hid, K * latent_dim) # hidden to mu for all K modes 

            if cov_type == "diag": # diagonal covariance
                self.logsig_head = nn.Linear(hid, K * latent_dim)
            elif cov_type == "lowrank": # low-rank + diag covariance
                self.logsig_head = nn.Linear(hid, K * latent_dim)
                self.U_head = nn.Linear(hid, K * latent_dim * cov_rank)
            elif cov_type == "full": # full covariance via Cholesky factorization
                self.L_head = nn.Linear(hid, K * latent_dim * latent_dim)
            else:
                raise ValueError("cov_type must be diag/full/lowrank")
        else: # if the omega is x-independent
            self.pi_logits = nn.Parameter(torch.zeros(K)) # pi itself is a parameter
            self.mu = nn.Parameter(torch.zeros(K, latent_dim)) # mu for all k modes them self are parameters
            if cov_type == "diag":
                self.log_sigma = nn.Parameter(torch.zeros(K, latent_dim))
            elif cov_type == "lowrank":
                self.log_sigma = nn.Parameter(torch.zeros(K, latent_dim))
                self.U = nn.Parameter(torch.zeros(K, latent_dim, cov_rank))
            elif cov_type == "full":
                self.L_raw = nn.Parameter(torch.zeros(K, latent_dim, latent_dim))

    def forward(self, feat=None):
        if self.xdep: # x-dependent case. feat = encoder(x)
            h = self.trunk(feat) # of size (B, hid_dim)
            pi = torch.softmax(self.pi_head(h), dim=-1) # pi is of size [B, K], self.pi_head(h) is of [B, K]
            B = h.size(0) # batch size
            mu = self.mu_head(h).view(B, self.K, self.D) # mu is of size [B, K, D], D (latent dim) is CxHxW if no decoder
            if self.cov_type == "diag":
                sigma = torch.exp(self.logsig_head(h).view(B, self.K, self.D)).clamp_min(1e-6) # for numerical stability
                return pi, mu, sigma
            elif self.cov_type == "full":
                L_raw = self.L_head(h).view(B, self.K, self.D, self.D)
                L = torch.tril(L_raw)
                diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-6
                L = L.clone()
                idx = torch.arange(self.D, device=L.device)
                L[..., idx, idx] = diag
                return pi, mu, L
            else:
                U = self.U_head(h).view(B, self.K, self.D, self.r)
                sigma = torch.exp(self.logsig_head(h).view(B, self.K, self.D)).clamp_min(1e-6)
                return pi, mu, (U, sigma)
        else: # x-independent case (no feature input needed, here feat is only for batch size)
            B = feat.size(0) if feat is not None else 1
            pi = self.pi_logits.softmax(-1).expand(B, -1) # [B, K]
            mu = self.mu.unsqueeze(0).expand(B, -1, -1) # [B, K, D]
            if self.cov_type == "diag":
                sigma = torch.exp(self.log_sigma).clamp_min(1e-6).unsqueeze(0).expand(B, -1, -1) # [B, K, D]
                return pi, mu, sigma
            elif self.cov_type == "full":
                L_raw = self.L_raw.unsqueeze(0).expand(B, -1, -1, -1)
                L = torch.tril(L_raw) # lower triangle matrix but only a view
                diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-6
                L = L.clone() # the new lower triangular matrix, avoiding inplace problem
                idx = torch.arange(self.D, device=L.device)
                L[..., idx, idx] = diag
                return pi, mu, L
            else:
                U = self.U.unsqueeze(0).expand(B, -1, -1, -1) # [B, K, D, r]
                sigma = torch.exp(self.log_sigma).clamp_min(1e-6).unsqueeze(0).expand(B, -1, -1) # [B, K, D]
                return pi, mu, (U, sigma)

    # -------------------- GMM builder --------------------
    def mixture(self, pi, mu, cov, cov_type="diag"):
        mix = Categorical(pi)
        if cov_type == "diag":
            comp = Independent(Normal(mu, cov), 1) # the last dim is event dim
        elif cov_type == "full":
            comp = MultivariateNormal(loc=mu, scale_tril=cov)
        elif cov_type == "lowrank":
            U, sigma = cov # U: (B,K,D,r), sigma: (B,K,D)
            B, K, D, _ = U.shape
            eye = torch.eye(D, device=U.device).view(1, 1, D, D) # identity matrix of (1,1,D,D)
            cov_mat = U @ U.transpose(-1, -2) + (sigma**2).unsqueeze(-1)*eye + 1e-5*eye
            comp = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
        else:
            raise ValueError("cov_type must be diag/full/lowrank")
        return MixtureSameFamily(mix, comp)

    # -------------------- PR loss (with optional decoder) --------------------
    def pr_loss(self, net, pi, mu, cov, x, y, decoder, gamma=8/255, S=1, norm_type="linf",
                cov_type="diag", use_decoder=True, out_shape=None,
                loss_variant: str = "softplus",   # {"softplus","clamped","ce"}
                margin_clip: float = 5.0,         # used by "clamped"
                softplus_beta: float = 1.0):      # smoothness for "softplus"

        B, C, H, W = x.shape
        K, D = mu.size(1), mu.size(2)
        total = 0.0

        for k in range(K):
            # ---- sample latent z_k ----
            if cov_type == "diag":
                sigma = cov[:, k, :]
                z = torch.randn(S, B, D, device=x.device)
                lat = mu[:, k, :].unsqueeze(0) + sigma.unsqueeze(0) * z

            elif cov_type == "full":
                L = cov[:, k, :, :] # is of size [B, D, D]
                z = torch.randn(S, B, D, device=x.device).unsqueeze(-1) # [S, B, D, 1]
                # lat = mu + L @ z
                # lat = mu[:, k, :].unsqueeze(0) + torch.einsum('bde,sbe->sbd', L, z) # [D, D] x [D] -> [D]
                lat = mu[:, k, :].unsqueeze(0) + (L.unsqueeze(0) @ z).squeeze(-1)

            else:  # lowrank
                U, sigma = cov
                Uk = U[:, k, :, :] # is of size [B, D, r]
                sigk = sigma[:, k, :] # is of size [B, D]
                z1 = torch.randn(S, B, Uk.size(-1), device=x.device) # is of [S, B, r]
                z2 = torch.randn(S, B, D, device=x.device) # is of [S, B, D]
                lat = (mu[:, k, :].unsqueeze(0) # is of [1, B, D]
                       + torch.einsum('sbr,bdr->sbd', z1, Uk) # [S, B, r] x [B, D, r] -> [S, B, D]
                       + sigk.unsqueeze(0) * z2) # sigk.unsqueeze(0) is of [1, B, D]

            # ---- latent -> unconstrained noise u ----
            if use_decoder:
                u = decoder(lat.view(S * B, D))
            else:
                assert D == C * H * W, f"D={D} must equal C*H*W={C*H*W} when --use_decoder=False"
                u = lat.reshape(S * B, C, H, W)

            # ---- project to norm ball ----
            eps = g_ball(u, gamma=gamma, norm_type=norm_type)

            # ---- replicate inputs/labels for S samples ----
            x_rep = x.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S * B, C, H, W)
            y_rep = y.repeat(S)

            # ---- logits (centered for stability) ----
            logits = net(x_rep + eps)
            logits = logits - logits.max(dim=1, keepdim=True).values  # shift-invariant; avoids huge magnitudes

            if loss_variant == "ce":
                # consider - CE directly    
                L_phi = -F.cross_entropy(logits, y_rep, reduction="none").view(S, B).mean(0)

            else:

                # ---- stable worst-case surrogate ----
                # Correct logit and max of "other" logits (mask with -inf, not -1e9)
                correct = logits.gather(1, y_rep.view(-1, 1)).squeeze(1) # the logits of ground truth, size of (S*B,)
                mask_other = F.one_hot(y_rep, num_classes=logits.size(1)).bool() # mask others = false, (S*B,C)
                max_other = logits.masked_fill(mask_other, -float('inf')).max(1).values # fill the ground truth logit with -inf, then take max

                margin = correct - max_other  # >0 means correct by margin

                if loss_variant == "clamped":
                    # bounded surrogate: avoid -inf by clamping the margin range
                    L_wc = torch.clamp(margin, min=-margin_clip, max=margin_clip)
                else:
                    # "softplus" (smooth hinge): log(1 + exp(-margin))
                    # large positive margins → ~0; large negative margins → linear, stable
                    L_wc = F.softplus(margin, beta=softplus_beta)

                L_phi = L_wc.view(S, B).mean(0)  # average over S

            # ---- mixture weighting ----
            total = total + pi[:, k] * L_phi

        return total.mean()

    # -------------------- Train φ --------------------
    def fit(self, model, feat_extractor, loader, args, device, out_shape, batch_indices=None): # out_shape: (C,H,W)
        # pick encoder
        ext_encoder, ext_dim = build_encoder(args.encoder_backend, out_shape, device,
                                             ckpt=args.encoder_ckpt, freeze=args.freeze_encoder)
        if args.encoder_backend == "classifier": # if reusing classifier's feat_extractor
            encoder = feat_extractor
            feat_dim = infer_feat_dim(encoder, out_shape)
        else:
            encoder = ext_encoder
            feat_dim = ext_dim

        # decoder or pixel
        if args.use_decoder: # use a decoder to reduce the latent dimension
            decoder, eff_latent = load_decoder_backend(args.decoder_backend, args.latent_dim,
                                                       out_shape, device, args.freeze_decoder,
                                                       args.gan_class, args.gan_truncation)
        else: # if no decoder needed, latent dim must be C*H*W (effective latent dim)
            decoder, eff_latent = None, out_shape[0]*out_shape[1]*out_shape[2]

        # NOTE: you instantiated this class with latent_dim == eff_latent
        assert eff_latent == self.D, f"latent dim mismatch: decoder/pixel {eff_latent} vs GMM.D {self.D}"

        params = list(self.parameters())
        if (args.encoder_backend != "classifier") and (not args.freeze_encoder): # train the external encoder if not frozen
            params += list(encoder.parameters())
        if args.use_decoder and (not args.freeze_decoder):
            params += list(decoder.parameters())

        opt = torch.optim.Adam(params, lr=args.lr)

        # --- per-batch loss history ---
        loss_hist = {}  # it -> [loss at ep1, loss at ep2, ...]

        for ep in range(1, args.epochs + 1):
            for it, (x, y, _) in enumerate(loader):
                if (batch_indices is not None) and (it not in batch_indices): # target specific batches only
                    continue

                x, y = x.to(device), y.to(device)

                # --- NEW: mask only correctly classified samples ---
                with torch.no_grad():
                    logits_clean = model(x)
                    pred_clean = logits_clean.argmax(1)
                    mask = (pred_clean == y)

                if mask.sum() == 0:
                    print(f"[ep{ep} it{it}] skipped (0 clean-correct / {y.numel()})")
                    continue  # skip this batch entirely

                x, y = x[mask], y[mask]  # keep only clean-correct ones

                if self.xdep:
                    feat = encoder(x) # frozen feature extractor if reuse the classifier's features
                else:
                    feat = torch.zeros(x.size(0), feat_dim, device=device) # [B, feat_dim]

                pi, mu, cov = self(feat) # cov is sigma or L or (U,sigma)
                loss = self.pr_loss(model, pi, mu, cov, x, y, decoder,
                                    gamma=args.gamma, S=args.mc, norm_type=args.norm,
                                    cov_type=self.cov_type, use_decoder=args.use_decoder, out_shape=out_shape,
                                    loss_variant="ce", softplus_beta=1.0  # smoother
                                    )

                opt.zero_grad()
                loss.backward()
                opt.step()

                # record
                loss_hist.setdefault(it, []).append(float(loss.item()))

                if it % 1 == 0:
                    print(f"[ep{ep} it{it}] loss={loss.item():.4f}  (used {y.numel()} clean-correct)")
                    # break  # debug only one batch per epoch

        return encoder, decoder, loss_hist


    # -------------------- Save / Load package --------------------
    def save_package(self,
                     filepath: str,
                     *,
                     encoder: nn.Module = None,
                     decoder: nn.Module = None,
                     args=None,
                     extra_meta: dict = None):
        """
        Save a reusable package to `filepath` that contains:
          - gmm_state:            self.state_dict()
          - gmm_config:           structure hyperparams (K, D, xdep, cov_type, cov_rank, feat_dim)
          - encoder_info:         backend name, whether used/frozen, and (optionally) its state_dict
          - decoder_info:         backend name, whether used/frozen, and (optionally) its state_dict
          - args_snapshot:        a light snapshot of key args (just for reproducibility)
          - extra_meta:           any user-provided small dict to store

        NOTE:
          * We always log which encoder/decoder backend was used (even if not trainable),
            so later you can rebuild the same components.
          * If an encoder/decoder is trainable (has any requires_grad=True), we also save its weights.
        """
        pkg = {}

        # ---- core GMM ----
        pkg["gmm_state"] = self.state_dict()
        pkg["gmm_config"] = {
            "K": self.K,
            "D": self.D,
            "xdep": self.xdep,
            "cov_type": self.cov_type,
            "cov_rank": self.r,
            # feat_dim is needed to rebuild x-dependent trunk
            "feat_dim": getattr(self, "trunk")[0].in_features if self.xdep else 0,
        }

        # ---- encoder info ----
        enc_info = {
            "backend": getattr(args, "encoder_backend", None),
            "freeze_encoder": bool(getattr(args, "freeze_encoder", False)),
            "used": encoder is not None,
            "state_dict": None,
        }
        if encoder is not None:
            # if any param is trainable, we save weights; otherwise仅记录 backend 即可
            trainable = any(p.requires_grad for p in encoder.parameters())
            if trainable:
                enc_info["state_dict"] = encoder.state_dict()
            enc_info["trainable"] = trainable
        pkg["encoder_info"] = enc_info

        # ---- decoder info ----
        dec_info = {
            "backend": getattr(args, "decoder_backend", None),
            "use_decoder": bool(getattr(args, "use_decoder", False)),
            "freeze_decoder": bool(getattr(args, "freeze_decoder", False)),
            "latent_dim": getattr(args, "latent_dim", None),
            "gan_class": getattr(args, "gan_class", None),
            "gan_truncation": getattr(args, "gan_truncation", None),
            "used": decoder is not None,
            "state_dict": None,
        }
        if decoder is not None:
            trainable = any(p.requires_grad for p in decoder.parameters())
            if trainable:
                # Only save the *learnable* part — e.g. your ConvDecoder; for BigGAN (frozen) no need
                dec_info["state_dict"] = decoder.state_dict()
            dec_info["trainable"] = trainable
        pkg["decoder_info"] = dec_info

        # ---- args snapshot (optional) ----
        if args is not None:
            keys = [
                "dataset","arch","norm","gamma","mc","num_modes","cov_type","cov_rank",
                "xdep","use_decoder","encoder_backend","decoder_backend",
                "freeze_encoder","freeze_decoder","latent_dim",
                "gan_class","gan_truncation"
            ]
            pkg["args_snapshot"] = {k: getattr(args, k, None) for k in keys}
        else:
            pkg["args_snapshot"] = None

        # ---- user meta ----
        pkg["extra_meta"] = extra_meta or {}

        torch.save(pkg, filepath)
        print(f"[GMM] package saved to: {filepath}")

    @staticmethod
    def load_package(filepath: str,
                     device: torch.device,
                     *,
                     build_encoder_fn,
                     load_decoder_backend_fn,
                     out_shape=None):
        """
        Load a package saved by `save_package`, and rebuild:
          - gmm:       GMM module with loaded weights
          - encoder:   either rebuilt from backend + (optional) loaded weights, or None
          - decoder:   either rebuilt from backend + (optional) loaded weights, or None
          - meta:      the raw dict with config and args_snapshot

        Args:
          build_encoder_fn:      callable(backend, img_shape, device, ckpt="", freeze=True) -> (encoder, feat_dim)
          load_decoder_backend_fn: callable(backend, latent_dim, out_shape, device, freeze, gan_class, gan_trunc) -> (decoder, eff_latent)
          out_shape:             necessary to rebuild decoder or check pixel path when use_decoder=False
        """
        pkg = torch.load(filepath, map_location="cpu")
        cfg = pkg["gmm_config"]

        # ---- rebuild GMM skeleton ----
        feat_dim = cfg["feat_dim"] if cfg["xdep"] else 0
        gmm = GMM(
            feat_dim=feat_dim,
            K=cfg["K"],
            latent_dim=cfg["D"],
            xdep=cfg["xdep"],
            cov_type=cfg["cov_type"],
            cov_rank=cfg["cov_rank"],
        ).to(device)
        gmm.load_state_dict(pkg["gmm_state"])
        gmm.eval()

        # ---- rebuild encoder (if used) ----
        enc = None
        enc_info = pkg.get("encoder_info", {}) or {}
        enc_backend = enc_info.get("backend", None)
        if cfg["xdep"]:
            if enc_backend is None:
                raise RuntimeError("Missing encoder backend in package for x-dependent GMM.")
            # Rebuild encoder via factory; freeze according to saved flag
            freeze_enc = bool(enc_info.get("freeze_encoder", True))
            enc, feat_dim_chk = build_encoder_fn(enc_backend, out_shape, device, ckpt="", freeze=freeze_enc)
            # optional: load saved weights (if trainable when saved)
            sd = enc_info.get("state_dict", None)
            if sd is not None:
                enc.load_state_dict(sd, strict=False)
            if freeze_enc:
                enc.eval()
        # else: x-independent → no encoder needed

        # ---- rebuild decoder (if used) ----
        dec = None
        dec_info = pkg.get("decoder_info", {}) or {}
        if dec_info.get("use_decoder", False):
            if out_shape is None:
                raise RuntimeError("`out_shape` is required to rebuild decoder.")
            dec_backend = dec_info.get("backend", None)
            latent_dim = dec_info.get("latent_dim", cfg["D"])
            gan_class = dec_info.get("gan_class", 207)
            gan_trunc = dec_info.get("gan_truncation", 0.5)
            freeze_dec = bool(dec_info.get("freeze_decoder", True))
            dec, eff_latent = load_decoder_backend_fn(dec_backend, latent_dim, out_shape, device, freeze_dec,
                                                      gan_class, gan_trunc)
            sd = dec_info.get("state_dict", None)
            if sd is not None:
                dec.load_state_dict(sd, strict=False)
            if freeze_dec:
                dec.eval()

        meta = {
            "gmm_config": cfg,
            "args_snapshot": pkg.get("args_snapshot"),
            "encoder_info": enc_info,
            "decoder_info": dec_info,
            "extra_meta": pkg.get("extra_meta"),
        }
        print(f"[GMM] package loaded from: {filepath}")

        return gmm, enc, dec, meta


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100","tinyimagenet"], default="cifar10")
    ap.add_argument("--arch", choices=["resnet18","resnet50","wide_resnet50_2","vgg16","densenet121","mobilenet_v3_large","efficientnet_b0","vit_b_16"], \
                    default="resnet18")
    ap.add_argument("--clf_ckpt", type=str, default="./model_zoo/trained_model/ResNets/resnet18_cifar10.pth", \
                    help="path to trained classifier checkpoint (required)")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--batch_idx", type=str, default="2",
                    help="Which batch indices to TRAIN and compute PR on (e.g., '0', '0,3,5-10'). Empty=all.")
    ap.add_argument("--viz_batch", type=int, default=2,
                    help="Batch index for the PCA viz (-1 means first batch).")

    # --- NEW: external encoder controls x-dependence ---
    ### I think it is good to use pretrained encoders and decoders for better representation power, 
    ### no need to train during fitting φ
    ap.add_argument("--encoder_backend", choices=["classifier","resnet18_imnet","vit_b_16_imnet","cnn_tiny"], \
                    default="classifier", help="choose external encoder to parameterize x-dependent pi/mu/sigma")
    ap.add_argument("--encoder_ckpt", default="", help="path to your pretrained encoder (optional)")
    ap.add_argument("--freeze_encoder", action="store_true", default=True, help="freeze the external encoder") # store false for test

    # Decoder control
    ap.add_argument("--use_decoder", action="store_true", default=False, \
                    help="use decoder to map latent->image noise; else direct pixel latent") # store false for test
    ap.add_argument("--decoder_backend", choices=["conv","biggan256"], default="conv")
    ap.add_argument("--freeze_decoder", action="store_true", default=True, help="freeze the decoder") # store false for test
    ap.add_argument("--gan_class", type=int, default=207)
    ap.add_argument("--gan_truncation", type=float, default=0.5)
    ap.add_argument("--latent_dim", type=int, default=64, help="latent dim (only used when --use_decoder)")

    # GMM settings
    ap.add_argument("--num_modes", type=int, default=3)
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default="lowrank")
    ap.add_argument("--cov_rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--gamma", type=float, default=8/255)
    ap.add_argument("--mc", type=int, default=1, \
                    help="MC samples per image per step")
    ap.add_argument("--xdep", default=False, action="store_true") # store true for test
    ap.add_argument("--norm", choices=["l2","linf"], default="linf")
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_sel = parse_batch_spec(args.batch_idx)
    # pr_sel    = parse_batch_spec(args.batch_idx)
    # viz_idx   = None if (args.viz_batch is None or args.viz_batch < 0) else args.viz_batch


    dataset, num_classes, out_shape = get_dataset(args.dataset, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, feat_extractor = build_model(args.arch, num_classes, device)

    # The downstream classifier and feat_extractor are frozen
    model = model.eval(); [p.requires_grad_(False) for p in model.parameters()] 
    feat_extractor = feat_extractor.eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()]


    # Load classifier ckpt
    if not args.clf_ckpt or not os.path.isfile(args.clf_ckpt):
        raise ValueError("You must provide --clf_ckpt pointing to a trained classifier on this dataset.")
    
    state = torch.load(args.clf_ckpt, map_location="cpu", weights_only=False) # map_location for debug on my laptop
    if "state_dict" in state: 
        # only leave the state dict in {"epoch": 10, "state_dict": model.state_dict(), "optimizer": opt.state_dict()}
        state = state["state_dict"]     
    state = {k.replace("module.",""): v for k,v in state.items()} # in case of dataparallel, trained with multi-gpu
    missing, unexpected = model.load_state_dict(state, strict=False) # check loading 
    print(f"[clf] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    # Check clean accuracy
    model.eval()
    eval_acc(model, dataset, device) 

    # Train φ
    head, encoder, decoder, loss_hist = head.fit(
        model, feat_extractor, loader, args, device, out_shape,
        batch_indices=train_sel
    )

    # Show convergence of batches across epochs
    plot_convergence(loss_hist, save_dir="viz", max_batches=5)


    # save the fitted GMM
    save_path = os.path.join("checkpoints", "gmm_pkg.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    head.save_package(
        save_path,
        encoder=encoder if args.xdep else None,   # x-independent then None
        decoder=decoder if args.use_decoder else None,
        args=args,                                # record key configurations (backend/flags, etc.)
        extra_meta={"note": "trained on clean-correct only"}  # your notes
    )



if __name__=="__main__":
    main()
