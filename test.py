import os
import argparse
import math
from unittest import loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily, MultivariateNormal
from attacks import WithIndex

# -------------------- Dataset utils --------------------
def get_dataset(name, root="./dataset", train=False):
    name = name.lower()
    if name == "cifar10":
        mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf))
        return ds, 10, (3,32,32)
    elif name == "cifar100":
        mean, std = (0.5071,0.4865,0.4409), (0.2673,0.2564,0.2762)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=tf))
        return ds, 100, (3,32,32)
    elif name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.MNIST(root=root, train=train, download=True, transform=tf))
        return ds, 10, (1,28,28)
    elif name == "tinyimagenet":
        mean, std = (0.4802,0.4481,0.3975), (0.2302,0.2265,0.2262)
        tf = T.Compose([T.Resize(64),T.CenterCrop(64),T.ToTensor(),T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.ImageFolder(os.path.join(root,"tiny-imagenet-200","val"), transform=tf))
        return ds, 200, (3,64,64)
    else:
        raise ValueError(f"Unknown dataset {name}")

def parse_batch_spec(spec):
    """
    Parse a batch selection string to a sorted set of ints.
    Examples:
      "" or None   -> None   (use all batches)
      "0"          -> {0}
      "1,3,7"      -> {1,3,7}
      "5-10"       -> {5,6,7,8,9,10}
      "0,4-6,12"   -> {0,4,5,6,12}
    """
    if spec is None or str(spec).strip() == "":
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            if a > b: a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return set(sorted(out))


# -------------------- Classifier factory (unchanged) --------------------
def build_model(arch: str, num_classes: int, device):
    arch = arch.lower()
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif arch == "resnet50":
        model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif arch == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif arch == "vgg16":
        model = torchvision.models.vgg16(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    elif arch == "densenet121":
        model = torchvision.models.densenet121(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(model.features, nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    elif arch == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    elif arch == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    elif arch == "vit_b_16":
        model = torchvision.models.vit_b_16(weights=None, num_classes=num_classes)
        class ViTFeat(nn.Module):
            def __init__(self, vit): super().__init__(); self.vit = vit
            def forward(self,x):
                x = self.vit._process_input(x)
                n = x.shape[0]
                cls_tok = self.vit.class_token.expand(n,-1,-1)
                x = torch.cat([cls_tok,x],dim=1)
                x = self.vit.encoder(x); x = x[:,0]
                return self.vit.ln(x) if hasattr(self.vit,"ln") else x
        feat_extractor = ViTFeat(model)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    
    # The downstream classifier and feat_extractor are frozen
    model = model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()] 
    feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()]
    
    return model, feat_extractor

@torch.no_grad()
def infer_feat_dim(fe: nn.Module, img_shape):
    C,H,W = img_shape
    dummy = torch.zeros(1, C, H, W, device=next(fe.parameters()).device)
    return fe(dummy).shape[-1]

# Evaluation 
@torch.no_grad()
def eval_acc(model, dataset, device):
    # load the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()

    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item() 
        total += y.numel()
    acc = correct / total
    print(f"[clf] accuracy={acc * 100:.2f}%"); return acc


# -------------------- External Encoder (NEW) --------------------
class TinyCNNEncoder(nn.Module):
    """一个轻量可训练的 Encoder（不依赖 ImageNet 预训练）"""
    def __init__(self, in_shape=(3,32,32), out_dim=256):
        super().__init__()
        C,H,W = in_shape
        ch = 32
        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 2, 1), nn.ReLU(),  # /2
            nn.Conv2d(ch, ch*2, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, 2, 1), nn.ReLU(),  # /4
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(ch*2, out_dim), nn.ReLU()
        )
        self.out_dim = out_dim
    def forward(self, x): return self.net(x)

def build_encoder(backend: str, img_shape, device, ckpt="", freeze=True):
    """
    返回: encoder(nn.Module), feat_dim(int)
    backend:
      - 'classifier': 复用分类器的 feat_extractor
      - 'resnet18_imnet': torchvision resnet18 (ImageNet 预训练) 去掉 fc
      - 'vit_b_16_imnet': torchvision vit_b_16 (ImageNet 预训练) 取 CLS
      - 'cnn_tiny': 轻量自定义 CNN 编码器（可训练）
    """
    backend = backend.lower()
    C,H,W = img_shape

    if backend == "classifier":
        # 由外层传入（我们在 fit_phi 里直接选用 feat_extractor）
        return None, None

    if backend == "resnet18_imnet":
        try:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = torchvision.models.resnet18(weights=None)
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        feat_dim = 512
    elif backend == "vit_b_16_imnet":
        try:
            vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        except Exception:
            vit = torchvision.models.vit_b_16(weights=None)
        class ViTFeat(nn.Module):
            def __init__(self, m): super().__init__(); self.m=m
            def forward(self,x):
                x=self.m._process_input(x); n=x.shape[0]
                cls=self.m.class_token.expand(n,-1,-1); x=torch.cat([cls,x],dim=1)
                x=self.m.encoder(x); x=x[:,0]; return self.m.ln(x) if hasattr(self.m,"ln") else x
        encoder = ViTFeat(vit)
        feat_dim = vit.hidden_dim
    elif backend == "cnn_tiny":
        encoder = TinyCNNEncoder(in_shape=img_shape, out_dim=256)
        feat_dim = 256
    else:
        raise ValueError("encoder_backend must be one of {'classifier','resnet18_imnet','vit_b_16_imnet','cnn_tiny'}")

    encoder = encoder.to(device)
    # 可选加载你自己的预训练 ckpt
    if ckpt and os.path.isfile(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
            state = {k.replace("module.",""): v for k,v in state.items()}
            encoder.load_state_dict(state, strict=False)
            print("[info] external encoder ckpt loaded.")
        except Exception as e:
            print(f"[warn] failed to load encoder ckpt: {e}")
    if freeze:
        for p in encoder.parameters(): p.requires_grad_(False)
        encoder.eval()
        print("[info] external encoder frozen.")
    else:
        print("[info] external encoder will be trained.")
    return encoder, feat_dim

# -------------------- Decoder backends (keep) --------------------
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_shape=(3,32,32)):
        super().__init__()
        C,H,W = out_shape
        self.fc = nn.Sequential(nn.Linear(latent_dim,256), nn.ReLU(),
                                nn.Linear(256,128*4*4), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.Conv2d(16,C,3,1,1)
        )
        self.out_shape = out_shape
    def forward(self, z):
        B=z.size(0) # batch size
        h=self.fc(z).view(B,128,4,4) # h is of like [64, 128, 4, 4]
        u=self.deconv(h) # u is of [64, 3, 32, 32] of images 
        return F.interpolate(u, size=self.out_shape[1:], mode="bilinear", align_corners=False)

class BigGANDecoder(nn.Module):
    def __init__(self, out_shape=(3,32,32), gan_class=207, truncation=0.5, device="cpu"):
        super().__init__()
        self.out_shape=out_shape; self.truncation=truncation
        self.register_buffer("class_onehot", torch.zeros(1000)); self.class_onehot[gan_class]=1
        self.biggan=torch.hub.load('huggingface/pytorch-pretrained-BigGAN','biggan-deep-256',pretrained=True).to(device).eval()
        for p in self.biggan.parameters(): p.requires_grad_(False)
    def forward(self,z):
        B=z.size(0); class_vec=self.class_onehot.unsqueeze(0).expand(B,-1).to(z.device)
        u=self.biggan(z, class_vec, self.truncation)
        return F.interpolate(u, size=self.out_shape[1:], mode="bilinear", align_corners=False)

def load_decoder_backend(backend, latent_dim, out_shape, device, freeze, gan_class=207, gan_trunc=0.5):
    if backend == "conv":
        dec = ConvDecoder(latent_dim, out_shape).to(device)
        if freeze: [p.requires_grad_(False) for p in dec.parameters()]
        return dec, latent_dim
    elif backend == "biggan256":
        dec = BigGANDecoder(out_shape, gan_class, gan_trunc, device)
        if not freeze: [p.requires_grad_(True) for p in dec.biggan.parameters()]
        return dec, 128
    else:
        raise ValueError("unknown decoder_backend")

# ------------------ GMM Head (diag/full/lowrank) --------------------
class GMMHead(nn.Module):
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
                U = self.U.unsqueeze(0).expand(B, -1, -1, -1)
                sigma = torch.exp(self.log_sigma).clamp_min(1e-6).unsqueeze(0).expand(B, -1, -1)
                return pi, mu, (U, sigma)

# -------------------- GMM builder --------------------
def build_gmm(pi, mu, cov, cov_type="diag"):
    mix = Categorical(pi)
    if cov_type == "diag":
        comp = Independent(Normal(mu, cov), 1) # the last dim is event dim
    elif cov_type == "full":
        comp = MultivariateNormal(loc=mu, scale_tril=cov)
    elif cov_type == "lowrank":
        U, sigma = cov
        B, K, D, _ = U.shape
        eye = torch.eye(D, device=U.device).view(1, 1, D, D)
        cov_mat = U @ U.transpose(-1, -2) + (sigma**2).unsqueeze(-1)*eye + 1e-5*eye
        comp = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    else:
        raise ValueError("cov_type must be diag/full/lowrank")
    return MixtureSameFamily(mix, comp)

# -------------------- g_B --------------------
def g_ball(u, gamma, norm_type):
    if norm_type == "linf":
        return gamma * u.tanh()
    if norm_type == "l2": # project onto l2 ball instead of tanh for stability
        flat = u.view(u.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        return (gamma * flat / norm).view_as(u)
    raise ValueError

# -------------------- PR loss (with optional decoder) --------------------
def PR_loss(net, pi, mu, cov, x, y, decoder, gamma=8/255, S=1, norm_type="linf",
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
            L = cov[:, k, :, :]
            z = torch.randn(S, B, D, device=x.device).unsqueeze(-1)
            lat = mu[:, k, :].unsqueeze(0) + (L.unsqueeze(0) @ z).squeeze(-1)

        else:  # lowrank
            U, sigma = cov
            Uk = U[:, k, :, :]
            sigk = sigma[:, k, :]
            z1 = torch.randn(S, B, Uk.size(-1), device=x.device)
            z2 = torch.randn(S, B, D, device=x.device)
            lat = (mu[:, k, :].unsqueeze(0)
                   + torch.einsum('sbr,bdr->sbd', z1, Uk)
                   + sigk.unsqueeze(0) * z2)

        # ---- latent -> unconstrained noise u ----
        if use_decoder:
            u = decoder(lat.view(S * B, D))
        else:
            assert D == C * H * W, f"D={D} must equal C*H*W={C*H*W} when --use_decoder=False"
            u = lat.view(S * B, C, H, W)

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
def fit_phi(model, feat_extractor, loader, args, device, out_shape, batch_indices=None): # out_shape: (C,H,W)
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

    head = GMMHead(feat_dim, K=args.num_modes, latent_dim=eff_latent,
                   xdep=args.xdep, cov_type=args.cov_type, cov_rank=args.cov_rank).to(device)

    params = list(head.parameters())
    if (args.encoder_backend != "classifier") and (not args.freeze_encoder):
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

            if args.xdep:
                feat = encoder(x) # frozen feature extractor if reuse the classifier's features
            else:
                feat = torch.zeros(x.size(0), feat_dim, device=device) # [B, feat_dim]

            pi, mu, cov = head(feat) # cov is sigma or L or (U,sigma)
            loss = PR_loss(model, pi, mu, cov, x, y, decoder,
                           gamma=args.gamma, S=args.mc, norm_type=args.norm,
                           cov_type=args.cov_type, use_decoder=args.use_decoder, out_shape=out_shape,
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

    return head, encoder, decoder, loss_hist

def print_convergence(loss_hist, max_batches=10):
    """
    Print per-batch losses across epochs for the first few batches.
    """
    print("\n=== Per-batch convergence (loss across epochs) ===")
    for it in sorted(loss_hist.keys())[:max_batches]:
        seq = loss_hist[it]
        seq_str = ", ".join(f"{v:.4f}" for v in seq)
        print(f"batch {it:03d}: [{seq_str}]")

def plot_convergence(loss_hist, save_dir="viz", max_batches=5):
    """
    Plot loss vs epoch for the first few batches.
    """
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    for it in sorted(loss_hist.keys())[:max_batches]:
        y = loss_hist[it]
        x = list(range(1, len(y)+1))
        plt.plot(x, y, marker='o', label=f"batch {it}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Per-batch convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_per_batch.png"))
    plt.close()
    print(f"[viz] saved: {os.path.join(save_dir, 'convergence_per_batch.png')}")


# -------------------- Visualization --------------------

def cov_from_param(cov, cov_type):
    """Return Sigma: (B,K,D,D) from parameterization."""
    if cov_type == "diag":
        sigma = cov  # (B,K,D)
        B, K, D = sigma.shape
        eye = torch.eye(D, device=sigma.device).view(1,1,D,D)
        return (sigma**2).unsqueeze(-1) * eye  # (B,K,D,D)
    elif cov_type == "full":
        L = cov  # (B,K,D,D)
        return L @ L.transpose(-1, -2)
    else:  # lowrank
        U, sigma = cov  # U:(B,K,D,r), sigma:(B,K,D)
        B, K, D, _ = U.shape
        eye = torch.eye(D, device=U.device).view(1,1,D,D)
        return U @ U.transpose(-1, -2) + (sigma**2).unsqueeze(-1) * eye + 1e-6 * eye

def pca_2d(X):
    """
    X: (N,D) -> returns (mean, P, Y) with Y always (N,2)
    Handles D<2 or rank<2 by padding zeros.
    """
    N, D = X.shape
    Xm = X.mean(0, keepdim=True)
    Xc = X - Xm
    if D >= 2:
        try:
            U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
            P = Vt[:2, :]                      # (2,D)
            Y = Xc @ P.T                       # (N,2)
        except RuntimeError:
            # numerical fallback: one axis + zero pad
            P = torch.zeros(2, D, device=X.device); P[0, 0] = 1.0
            Y = torch.cat([Xc[:, :1], torch.zeros(N,1, device=X.device)], dim=1)
    else:
        # D == 1
        P = torch.zeros(2, D, device=X.device); P[0, 0] = 1.0
        Y = torch.cat([Xc, torch.zeros(N,1, device=X.device)], dim=1)
    return Xm.squeeze(0), P, Y

def ellipse_from_cov2(cov2, nsig=2.0):
    """cov2: (2,2) -> width,height,angle for a matplotlib Ellipse."""
    eigvals, eigvecs = torch.linalg.eigh(cov2)
    eigvals = eigvals.clamp_min(1e-12)
    radii = eigvals.sqrt() * nsig
    v = eigvecs[:, 1] if cov2.numel() == 4 else torch.tensor([1.0, 0.0], device=cov2.device)
    angle = float(torch.atan2(v[1], v[0]) * 180.0 / math.pi)
    # matplotlib Ellipse expects diameters
    return float(2*radii[-1]), float(2*radii[0]), angle


@torch.no_grad()
def viz_all(loader, head, encoder, decoder, build_gmm, g_ball, args, out_shape,
            save_dir="viz", viz_batch_idx=None):
    """
    Two figures:
      - pi_bar.png     : average mixture weights over a few batches
      - gmm_pca2d.png  : for ONE chosen batch (viz_batch_idx), plot μ (PCA->2D) and Σ ellipses
    viz_batch_idx:
      - None 或 <0 : 使用第一个 batch
      - >=0        : 使用该 batch 索引
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(head.parameters()).device
    C, H, W = out_shape

    # --------- (1) π bar across batches ----------
    max_batches = 50
    pis = []
    for i, (x, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        if args.xdep:
            feat = encoder(x)
        else:
            # x-independent：给一个与 head 期望宽度一致的零向量
            feat = torch.empty(x.size(0), 0, device=device)   # only batch size matters when xdep=False
        pi, _, _ = head(feat)  # (B,K)
        pis.append(pi.cpu())

    if len(pis) > 0:
        P = torch.cat(pis, 0)  # (N,K)
        avg_pi = P.mean(0).numpy()
        plt.figure(figsize=(max(6, 0.6 * len(avg_pi)), 3.5))
        plt.bar(range(len(avg_pi)), avg_pi)
        plt.xlabel("component k"); plt.ylabel("mean π_k")
        plt.title("Average mixture weights π")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pi_bar.png"))
        plt.close()
    else:
        print("[viz] warning: no batches found for π bar; skip.")

    # --------- (2) choose ONE batch for μ/Σ PCA plot ----------
    # 选择要可视化的 batch
    if viz_batch_idx is None or viz_batch_idx < 0:
        # 默认取第一个 batch
        x, _, _ = next(iter(loader))
    else:
        found = False
        for it, (xb, yb, _) in enumerate(loader):
            if it == viz_batch_idx:
                x = xb
                found = True
                break
        if not found:
            raise ValueError(f"viz_batch_idx={viz_batch_idx} not found in loader.")

    x = x[:1].to(device)  # 只用一张图来条件化
    if args.xdep:
        feat = encoder(x)
    else:
        feat = torch.empty(1, 0, device=device)   # only batch size matters when xdep=False

    pi, mu, cov = head(feat)                # shapes: pi(1,K), mu(1,K,D), cov depends
    Sigma = cov_from_param(cov, args.cov_type)[0]  # (K,D,D)  -> 真正的协方差矩阵

    K, D = mu.size(1), mu.size(2)
    muK = mu[0]                              # (K,D)

    # PCA using component means (safe to 2D)
    _, Pproj, Y = pca_2d(muK)                # Pproj: (2,D), Y: (K,2)
    Pm = Pproj.to(muK.device)

    # Project Σ to 2D: cov2[k] = P Σ_k P^T
    cov2 = torch.einsum('pd,kde,qd->kpq', Pm, Sigma, Pm)  # (K,2,2)

    # Plot
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(6, 6))
    Ynp = Y.detach().cpu().numpy()
    if Ynp.shape[1] == 1:  # 保险：如只剩 1 维则补零
        import numpy as np
        Ynp = np.concatenate([Ynp, np.zeros((Ynp.shape[0], 1))], axis=1)

    pik = pi[0].detach().cpu().numpy()
    sizes = 300.0 * (pik / max(pik.max(), 1e-8))

    ax.scatter(Ynp[:, 0], Ynp[:, 1], s=sizes, alpha=0.8, edgecolors='k')
    for k in range(K):
        w, h, ang = ellipse_from_cov2(cov2[k])
        e = Ellipse((Ynp[k, 0], Ynp[k, 1]), width=w, height=h, angle=ang,
                    fill=False, lw=2, alpha=0.8)
        ax.add_patch(e)
        ax.text(Ynp[k, 0], Ynp[k, 1], f"{k}", fontsize=9, ha='center', va='center')

    ax.set_title("GMM in latent space (PCA to 2D): means & covariance ellipses")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gmm_pca2d.png"))
    plt.close()

    print(f"[viz] saved: {os.path.join(save_dir,'pi_bar.png')} and gmm_pca2d.png")

# ------------------- PR computation -------------------
@torch.no_grad()
def compute_pr_on_clean_correct(model, head, encoder, loader, args, out_shape,
                                decoder=None, S=100, batch_indices=None):
    """
    PR on clean-correct set:
      PR = E_{(x,y) ∈ CleanCorrect}  E_{eps~GMM_phi(.|x)} [ 1{ f(x+g_B(eps)) = y } ]
    Returns (pr, n_used, clean_acc).
    """
    device = next(head.parameters()).device
    C, H, W = out_shape

    total_used = 0          # number of clean-correct samples included
    pr_sum = 0.0            # sum of per-sample success probabilities
    clean_correct = 0
    total_seen = 0

    for it, (x, y, _) in enumerate(loader):
        if (batch_indices is not None) and (it not in batch_indices):
            continue

        x, y = x.to(device), y.to(device) # for debug: {2}: -0.7620, -0.8007
        B = x.size(0)

        # clean preds & mask of correct ones
        logits_clean = model(x)
        pred_clean = logits_clean.argmax(1)
        mask = (pred_clean == y)
        clean_correct += mask.sum().item()
        total_seen += B
        if mask.sum().item() == 0:
            continue  # nothing to evaluate in this batch

        x_sel = x[mask]
        y_sel = y[mask]
        n = x_sel.size(0)

        # x-dependent parameters
        if args.xdep:
            feat = encoder(x_sel)
        else:
            # any feat vector works when xdep=False; just match the expected width
            feat = torch.empty(n, 0, device=device) # only batch size matters when xdep=False

        pi, mu, cov = head(feat)                 # shapes (n,K,*)
        gmm = build_gmm(pi, mu, cov, cov_type=args.cov_type)

        # sample S per selected item → [S, B(corrected) ,D]
        z = gmm.sample((S,)) 

        # latent -> image noise
        if args.use_decoder:
            u = decoder(z.view(S * n, -1))
        else:
            assert z.size(-1) == C * H * W, \
                f"When --use_decoder=False, latent dim must be C*H*W ({C*H*W})."
            u = z.view(S * n, C, H, W)

        eps = g_ball(u, gamma=args.gamma, norm_type=args.norm)

        x_rep = x_sel.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S * n, C, H, W)
        y_rep = y_sel.repeat(S)

        logits = model(x_rep + eps)
        pred = logits.argmax(1)
        # per-sample success prob over S draws
        succ = (pred == y_rep).float().view(S, n).mean(0)  # (n,)

        pr_sum += succ.sum().item()
        total_used += n

    pr = pr_sum / max(1, total_used)
    clean_acc = clean_correct / max(1, total_seen)
    print(f"[PR@clean] used={total_used} / seen={total_seen} (clean acc={clean_acc*100:.2f}%), "
          f"S={S} → PR={pr:.4f}")
    return pr, total_used, clean_acc


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100","mnist","tinyimagenet"], default="cifar10")
    ap.add_argument("--arch", choices=["resnet18","resnet50","wide_resnet50_2","vgg16","densenet121","mobilenet_v3_large","efficientnet_b0","vit_b_16"], \
                    default="resnet18")
    ap.add_argument("--clf_ckpt", type=str, default="./model_zoo/trained_model/ResNets/resnet18_cifar10.pth", \
                    help="path to trained classifier checkpoint (required)")
    ap.add_argument("--device", default="cuda:1")

    ap.add_argument("--train_batches", type=str, default="2",
                    help="Which batch indices to TRAIN on (e.g., '0', '0,3,5-10'). Empty=all.")
    ap.add_argument("--pr_batches", type=str, default="2",
                    help="Which batch indices to compute PR on. Empty=all.")
    ap.add_argument("--viz_batch", type=int, default=2,
                    help="Batch index for the PCA viz (-1 means first batch).")

    # --- NEW: external encoder controls x-dependence ---
    ap.add_argument("--encoder_backend", choices=["classifier","resnet18_imnet","vit_b_16_imnet","cnn_tiny"], \
                    default="classifier", help="choose external encoder to parameterize x-dependent pi/mu/sigma")
    ap.add_argument("--encoder_ckpt", default="", help="path to your pretrained encoder (optional)")
    ap.add_argument("--freeze_encoder", action="store_true", help="freeze the external encoder") # store ture for test

    # Decoder control
    ap.add_argument("--use_decoder", action="store_true", default=False, \
                    help="use decoder to map latent->image noise; else direct pixel latent") # store false for test
    ap.add_argument("--decoder_backend", choices=["conv","biggan256"], default="conv")
    ap.add_argument("--freeze_decoder", action="store_true")
    ap.add_argument("--gan_class", type=int, default=207)
    ap.add_argument("--gan_truncation", type=float, default=0.5)

    # GMM + training
    ap.add_argument("--num_modes", type=int, default=10)
    ap.add_argument("--latent_dim", type=int, default=64, help="latent dim (only used when --use_decoder)")
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default="diag")
    ap.add_argument("--cov_rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--gamma", type=float, default=8/255)
    ap.add_argument("--mc", type=int, default=50, \
                    help="MC samples per image per step")
    ap.add_argument("--xdep", default=False, action="store_true") # store true for test
    ap.add_argument("--norm", choices=["l2","linf"], default="linf")
    ap.add_argument("--batch_size", type=int, default=64)

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_sel = parse_batch_spec(args.train_batches)
    pr_sel    = parse_batch_spec(args.pr_batches)
    viz_idx   = None if (args.viz_batch is None or args.viz_batch < 0) else args.viz_batch


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

    # Check clean accuracy
    model.eval()
    eval_acc(model, dataset, device) 

    # Train φ
    head, encoder, decoder, loss_hist = fit_phi(
        model, feat_extractor, loader, args, device, out_shape,
        batch_indices=train_sel
    )

    # Show convergence of batches across epochs
    print_convergence(loss_hist, max_batches=10)
    plot_convergence(loss_hist, save_dir="viz", max_batches=5)

    _ = compute_pr_on_clean_correct(
            model,
            head,
            encoder if args.xdep else (feat_extractor if args.encoder_backend=="classifier" else encoder),
            loader,
            args,
            out_shape,
            decoder=decoder,
            S=100,
            batch_indices=pr_sel
        )

    viz_all(loader, head,
            encoder if args.xdep else (feat_extractor if args.encoder_backend=="classifier" else encoder),
            decoder, build_gmm, g_ball, args, out_shape,
            save_dir="viz",
            viz_batch_idx=viz_idx)


if __name__=="__main__":
    main()
