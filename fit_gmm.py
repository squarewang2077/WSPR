import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
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
        if args.use_decoder and (not args.freeze_decoder): # almost never train the decoder
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


    @torch.no_grad()
    def compute_pr_on_clean_correct(
        self,
        model,                 # downstream classifier f(.)
        loader,                # dataloader over (x,y, ...)
        out_shape,             # (C,H,W)
        *,
        encoder=None,          # encoder used only when self.xdep=True
        decoder=None,          # decoder if use_decoder=True
        S: int = 100,          # MC samples per image
        gamma: float = 8/255,  # projection radius
        norm_type: str = "linf",
        use_decoder: bool = True,
        batch_indices=None
    ):
        """
        PR on clean-correct set:
        PR = E_{(x,y)∈CleanCorrect} E_{eps~GMM_phi(.|x)}[ 1{ f(x+g_B(eps)) = y } ]

        Returns (pr_mean, n_used, clean_acc).

        Notes:
        - Uses self.xdep/self.cov_type/self.D 等结构属性。
        - encoder 仅在 self.xdep=True 时需要；其输出维度应匹配 self.trunk 的输入维度。
        - 当 use_decoder=False 时, latent 维度必须等于 C*H*W。
        """
        device = next(self.parameters()).device
        C, H, W = out_shape

        total_used = 0          # # of clean-correct samples actually evaluated
        pr_sum = 0.0            # sum of per-sample success probabilities
        clean_correct = 0
        total_seen = 0

        for it, (x, y, *_) in enumerate(loader):
            if (batch_indices is not None) and (it not in batch_indices):
                continue

            x = x.to(device); y = y.to(device)
            B = x.size(0)

            # ---- clean prediction & mask ----
            logits_clean = model(x)
            pred_clean = logits_clean.argmax(1)
            mask = (pred_clean == y)

            clean_correct += mask.sum().item()
            total_seen += B

            if mask.sum().item() == 0:
                continue  # no clean-correct in this batch

            x_sel = x[mask]
            y_sel = y[mask]
            n = x_sel.size(0)

            # ---- feature for x-dependent case ----
            if self.xdep:
                if encoder is None:
                    raise RuntimeError("GMM.compute_pr_on_clean_correct: self.xdep=True but encoder is None.")
                feat = encoder(x_sel)                    # (n, feat_dim)
            else:
                # x-independent: feat only provides batch size
                feat = torch.empty(n, 0, device=device)

            # ---- GMM parameters & mixture ----
            pi, mu, cov = self(feat) #.forward; shapes (n,K,*)
            gmm = self.mixture(pi, mu, cov, cov_type=self.cov_type)

            # ---- sample S per item: z ~ q(z|x) ----
            # shape -> (S, n, D), then flatten to (S*n, D) for decoding
            z = gmm.sample((S,))                         # (S, n, D)

            # ---- z -> u -> eps = g_B(u) ----
            if use_decoder and (decoder is not None):
                u = decoder(z.view(S * n, -1))           # (S*n, C,H,W) (your decoder should output images)
                if u.dim() == 2:                         # fallback in case decoder returns (N,D)
                    u = u.view(S * n, C, H, W)
            else:
                assert z.size(-1) == C * H * W, \
                    f"When use_decoder=False, latent dim must be C*H*W ({C*H*W}), got {z.size(-1)}."
                u = z.view(S * n, C, H, W)

            eps = g_ball(u, gamma=gamma, norm_type=norm_type)

            # ---- replicate inputs/labels for S MC samples ----
            x_rep = x_sel.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S * n, C, H, W)
            y_rep = y_sel.repeat(S)

            # ---- evaluate model ----
            logits = model(x_rep + eps)
            pred = logits.argmax(1)

            # per-sample success prob across S draws
            succ = (pred == y_rep).float().view(S, n).mean(0)  # (n,)
            pr_sum += succ.sum().item()
            total_used += n

        pr_mean = pr_sum / max(1, total_used)
        clean_acc = clean_correct / max(1, total_seen)

        print(f"[PR@clean] used={total_used} / seen={total_seen} "
            f"(clean acc={clean_acc*100:.2f}%), S={S} → PR={pr_mean:.4f}")

        return pr_mean, total_used, clean_acc



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

def _slug_gamma(g):
    # make gamma filename-safe, e.g. 0.03137255 -> 0p0314
    return f"{g:.4f}".replace('.', 'p')


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100","tinyimagenet"], default="cifar10")
    ap.add_argument("--arch", choices=["resnet18","resnet50","wide_resnet50_2","vgg16","densenet121","mobilenet_v3_large","efficientnet_b0","vit_b_16"], \
                    default="resnet18")
    ap.add_argument("--clf_ckpt", type=str, default="./model_zoo/trained_model/sketch/resnet18_cifar10.pth", \
                    help="path to trained classifier checkpoint (required)")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--batch_idx", type=str, default="",
                    help="Which batch indices to TRAIN and compute PR on (e.g., '0', '0,3,5-10'). Empty=all.")
    ap.add_argument("--viz_batch", type=int, default=-1,
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
    ap.add_argument("--decoder_backend", choices=["conv","biggan256"], default="biggan256", \
                    help="choose decoder backend if --use_decoder")
    ap.add_argument("--freeze_decoder", action="store_true", default=True, help="freeze the decoder") # store false for test
    ap.add_argument("--gan_class", type=int, default=207)
    ap.add_argument("--gan_truncation", type=float, default=0.5)
    ap.add_argument("--latent_dim", type=int, default=64, help="latent dim (only used when --use_decoder)")

    # GMM settings
    ap.add_argument("--num_modes", type=int, default=20) # plan to try 1,3,7,12,20 but for small model 
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default="diag")
    ap.add_argument("--cov_rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200) # 3 epochs for debug
    ap.add_argument("--lr", type=float, default=2e-2) # 2e-2 when x independent, 5e-4 when x-dependent
    ap.add_argument("--gamma", type=float, default=1/255) # 8/255 for Linf, 0.5 for L2
    ap.add_argument("--mc", type=int, default=20, \
                    help="MC samples per image per step")
    ap.add_argument("--xdep", default=False, action="store_true") # store true for test
    ap.add_argument("--norm", choices=["l2","linf"], default="linf")
    ap.add_argument("--batch_size", type=int, default=128) 



    args = ap.parse_args()
    cfg_str = f"CE_{args.arch}_{args.dataset}_cov({args.cov_type})_L({args.norm}_{_slug_gamma(args.gamma)})_K({args.num_modes})"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_sel = parse_batch_spec(args.batch_idx)
    # pr_sel    = parse_batch_spec(args.batch_idx)
    # viz_idx   = None if (args.viz_batch is None or args.viz_batch < 0) else args.viz_batch


    dataset, num_classes, out_shape = get_dataset(args.dataset, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, feat_extractor = build_model(args.arch, num_classes, device)

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

    # The downstream classifier and feat_extractor are frozen
    model = model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()] 
    feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()] # for safety

    # check that model and feat_extractor shares the same parameters if reuse classifier features
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
        
    # ---- Build GMM head BEFORE calling fit ----
    C, H, W = out_shape
    latent_eff = (args.latent_dim if args.use_decoder else C * H * W)

    # decide feat_dim for the GMM trunk
    if args.xdep:
        if args.encoder_backend == "classifier":
            # reuse classifier's feature extractor; probe its output dim
            feat_dim_for_gmm = infer_feat_dim(feat_extractor, out_shape)
        else:
            # probe external encoder's output dim once (we discard the temp encoder)
            _enc_probe, _enc_dim = build_encoder(
                args.encoder_backend, out_shape, device,
                ckpt=args.encoder_ckpt, freeze=args.freeze_encoder
            )
            feat_dim_for_gmm = _enc_dim
            del _enc_probe
    else:
        feat_dim_for_gmm = 0  # trunk not used

    # instantiate the GMM head
    head = GMM(
        feat_dim=feat_dim_for_gmm,     # must match the encoder output dim when xdep=True
        K=args.num_modes,
        latent_dim=latent_eff,         # must match decoder eff dim or C*H*W when no decoder
        xdep=args.xdep,
        cov_type=args.cov_type,
        cov_rank=args.cov_rank
    ).to(device)

    # ---------------- Train φ ----------------
    # GMM.fit() returns (encoder, decoder, loss_hist)
    encoder, decoder, loss_hist = head.fit(
        model,               # frozen classifier
        feat_extractor,      # frozen classifier's feat extractor (only used when encoder_backend='classifier')
        loader,
        args,
        device,
        out_shape,
        batch_indices=train_sel
    )

    # ---------------- Convergence plot ----------------
    plot_convergence(loss_hist, save_dir="viz", max_batches=5)

    # save the loss history as Pandas dataframe for future analysis
    loss_df = pd.DataFrame(dict([(f"batch_{k}", v) for k,v in loss_hist.items()]))
    os.makedirs("log/gmm_ckp/loss_hist", exist_ok=True)
    loss_df.to_csv(os.path.join("log/gmm_ckp/loss_hist", cfg_str + ".csv"), index=False)
    print(f"[save] loss history -> log/gmm_ckp/loss_hist/{cfg_str}.csv")


    # ---------------- Save the fitted GMM package ----------------
    save_root = os.path.join("./log/gmm_ckp", "x_dep" if args.xdep else "x_indep")
    os.makedirs(save_root, exist_ok=True)
    fname = f"gmm_{cfg_str}.pt"
    save_path = os.path.join(save_root, fname)

    head.save_package(
        save_path,
        encoder=encoder if args.xdep else None,         # x-independent → no encoder to store
        decoder=decoder if args.use_decoder else None,  # only store when using a decoder
        args=args,                                      # keep a light snapshot (backends/flags)
        extra_meta={"note": "trained on clean-correct only"}
    )
    print(f"[save] GMM package -> {save_path}")

    _ = compute_pr_on_clean_correct_old(
            model,
            head,
            encoder if args.xdep else (feat_extractor if args.encoder_backend == "classifier" else None),
            loader,
            args,
            out_shape,
            decoder=decoder,
            S=20,
            batch_indices=train_sel
        )

    # ---------------- (Optional) Visualization ----------------
    # viz_all(loader, head,
    #         encoder if args.xdep else (feat_extractor if args.encoder_backend=="classifier" else None),
    #         decoder, build_gmm, g_ball, args, out_shape,
    #         save_dir="viz",
    #         viz_batch_idx=None)



if __name__=="__main__":
    main()
