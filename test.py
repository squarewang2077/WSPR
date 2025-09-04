import os
import argparse
import math
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
    model = model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()]
    feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()]
    return model, feat_extractor

@torch.no_grad()
def infer_feat_dim(fe: nn.Module, img_shape):
    C,H,W = img_shape
    dummy = torch.zeros(1, C, H, W, device=next(fe.parameters()).device)
    return fe(dummy).shape[-1]

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
      - 'classifier': 复用分类器的 feat_extractor（与旧逻辑等价）
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
            state = torch.load(ckpt, map_location="cpu")
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
    def __init__(self, feat_dim, K=3, latent_dim=64, xdep=True, cov_type="diag", cov_rank=8):
        super().__init__()
        self.K, self.D, self.xdep = K, latent_dim, xdep
        self.cov_type, self.r = cov_type, cov_rank
        hid = 256
        if xdep:
            self.trunk = nn.Sequential(nn.Linear(feat_dim, hid), nn.ReLU())
            self.pi_head = nn.Linear(hid, K)
            self.mu_head = nn.Linear(hid, K * latent_dim)
            if cov_type == "diag":
                self.logsig_head = nn.Linear(hid, K * latent_dim)
            elif cov_type == "full":
                self.L_head = nn.Linear(hid, K * latent_dim * latent_dim)
            elif cov_type == "lowrank":
                self.U_head = nn.Linear(hid, K * latent_dim * cov_rank)
                self.logsig_head = nn.Linear(hid, K * latent_dim)
            else:
                raise ValueError("cov_type must be diag/full/lowrank")
        else:
            self.pi_logits = nn.Parameter(torch.zeros(K))
            self.mu = nn.Parameter(torch.zeros(K, latent_dim))
            if cov_type == "diag":
                self.log_sigma = nn.Parameter(torch.zeros(K, latent_dim))
            elif cov_type == "full":
                self.L_raw = nn.Parameter(torch.zeros(K, latent_dim, latent_dim))
            elif cov_type == "lowrank":
                self.U = nn.Parameter(torch.zeros(K, latent_dim, cov_rank))
                self.log_sigma = nn.Parameter(torch.zeros(K, latent_dim))

    def forward(self, feat=None):
        if self.xdep:
            h = self.trunk(feat)                       # (B,hid)
            pi = torch.softmax(self.pi_head(h), dim=-1)
            B = h.size(0)
            mu = self.mu_head(h).view(B, self.K, self.D)
            if self.cov_type == "diag":
                sigma = torch.exp(self.logsig_head(h).view(B, self.K, self.D)).clamp_min(1e-6)
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
        else:
            B = feat.size(0) if feat is not None else 1
            pi = self.pi_logits.softmax(-1).expand(B, -1)
            mu = self.mu.unsqueeze(0).expand(B, -1, -1)
            if self.cov_type == "diag":
                sigma = torch.exp(self.log_sigma).clamp_min(1e-6).unsqueeze(0).expand(B, -1, -1)
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
        comp = Independent(Normal(mu, cov), 1)
    elif cov_type == "full":
        comp = MultivariateNormal(loc=mu, scale_tril=cov)
    elif cov_type == "lowrank":
        U, sigma = cov
        B,K,D,_ = U.shape
        eye = torch.eye(D, device=U.device).view(1,1,D,D)
        cov_mat = U @ U.transpose(-1,-2) + (sigma**2).unsqueeze(-1)*eye + 1e-5*eye
        comp = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    else:
        raise ValueError("cov_type must be diag/full/lowrank")
    return MixtureSameFamily(mix, comp)

# -------------------- g_B --------------------
def g_ball(u, gamma, norm_type):
    if norm_type == "linf": return gamma * torch.tanh(u)
    if norm_type == "l2":
        flat = u.view(u.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True)
        dir = flat / (norm + 1e-12)
        r = gamma * torch.tanh(norm)
        return (dir * r).view_as(u)
    raise ValueError

# -------------------- Option A loss (with optional decoder) --------------------
def optionA_loss(net, pi, mu, cov, x, y, decoder, gamma=8/255, S=1, norm_type="linf",
                 cov_type="diag", use_decoder=True, out_shape=None):
    B, C, H, W = x.shape
    K, D = mu.size(1), mu.size(2)
    total = 0.0

    for k in range(K): #path sampling
        # latent sampling 
        if cov_type == "diag":
            sigma = cov[:, k, :]
            z = torch.randn(S, B, D, device=x.device) # sampling in standard Gaussian
            lat = mu[:, k, :].unsqueeze(0) + sigma.unsqueeze(0)*z
        elif cov_type == "full":
            L = cov[:, k, :, :] # Is Sigma = LL^T
            z = torch.randn(S, B, D, device=x.device).unsqueeze(-1)
            lat = mu[:, k, :].unsqueeze(0) + (L.unsqueeze(0) @ z).squeeze(-1) # mu_k + L@z
        else:
            U, sigma = cov
            Uk = U[:, k, :, :]
            sigk = sigma[:, k, :]
            z1 = torch.randn(S, B, Uk.size(-1), device=x.device)
            z2 = torch.randn(S, B, D, device=x.device)
            # einsum 更直观：s,b,r + b,d,r -> s,b,d
            lat = (mu[:, k, :].unsqueeze(0)
                   + torch.einsum('sbr,bdr->sbd', z1, Uk)
                   + sigk.unsqueeze(0)*z2)

        # latent -> 未约束像素噪声 u
        if use_decoder:
            u = decoder(lat.view(S*B, D))
        else:
            # 直接像素空间：D 必须等于 C*H*W
            u = lat.view(S*B, C, H, W)

        eps = g_ball(u, gamma=gamma, norm_type=norm_type)

        x_rep = x.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S*B, C, H, W)
        y_rep = y.repeat(S)
        logits = net(x_rep + eps)
        L_ce = F.cross_entropy(logits, y_rep, reduction="none").view(S,B).mean(0)
        total = total + pi[:, k] * L_ce

    return total.mean()

# -------------------- Train φ --------------------
def fit_phi(model, feat_extractor, loader, args, device, out_shape):
    # 选择 Encoder：若 encoder_backend='classifier'，就直接用分类器的 feature
    ext_encoder, ext_dim = build_encoder(args.encoder_backend, out_shape, device,
                                         ckpt=args.encoder_ckpt, freeze=args.freeze_encoder)
    if args.encoder_backend == "classifier":
        encoder = feat_extractor  # 复用分类器特征（旧逻辑）
        feat_dim = infer_feat_dim(encoder, out_shape)
    else:
        encoder = ext_encoder
        feat_dim = ext_dim

    # 选择 Decoder 或像素直连
    if args.use_decoder:
        decoder, eff_latent = load_decoder_backend(args.decoder_backend, args.latent_dim,
                                                   out_shape, device, args.freeze_decoder,
                                                   args.gan_class, args.gan_truncation)
    else:
        decoder, eff_latent = None, out_shape[0]*out_shape[1]*out_shape[2]  # C*H*W

    head = GMMHead(feat_dim, K=args.num_modes, latent_dim=eff_latent,
                   xdep=args.xdep, cov_type=args.cov_type, cov_rank=args.cov_rank).to(device)

    params = list(head.parameters())
    if (args.encoder_backend != "classifier") and (not args.freeze_encoder):
        params += list(encoder.parameters())
    if args.use_decoder and (not args.freeze_decoder):
        params += list(decoder.parameters())

    opt = torch.optim.Adam(params, lr=args.lr)

    for ep in range(1, args.epochs + 1):
        for it, (x, y, _) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if args.xdep:
                feat = encoder(x)
            else:    
                feat = torch.zeros(x.size(0), feat_dim, device=device)  # x-独立时占位
            
            pi, mu, cov = head(feat)
            loss = optionA_loss(model, pi, mu, cov, x, y, decoder,
                                gamma=args.gamma, S=args.mc, norm_type=args.norm,
                                cov_type=args.cov_type, use_decoder=args.use_decoder, out_shape=out_shape)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 5 == 0:
                print(f"[ep{ep} it{it}] loss={loss.item():.4f}")
    return head, encoder, decoder

# -------------------- Visualization --------------------
@torch.no_grad()
def viz_all(loader, head, encoder, decoder, build_gmm, g_ball, args, out_shape, save_dir="viz"):
    """
    Simple viz:
      1) pi_bar.png   – avg mixture weights over ~50 batches
      2) samples.png  – grid of perturbations sampled from the GMM for one image
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(head.parameters()).device
    C, H, W = out_shape

    # ---------- (1) Average mixture weights ----------
    max_batches = 50
    all_pi = []
    for i, (x, _, _) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device)
        if args.xdep:
            feat = encoder(x)
        else:
            # x-independent case: give a dummy feat vector of zeros (shape aligned with head)
            B = x.size(0)
            feat_dim = head.mu_head.in_features if head.xdep else head.mu.shape[-1]
            feat = torch.zeros(B, feat_dim, device=device)
        pi, _, _ = head(feat)      # (B,K)
        all_pi.append(pi.cpu())

    P = torch.cat(all_pi, dim=0)   # (N,K)
    avg = P.mean(0).numpy()

    plt.figure(figsize=(max(6, 0.6*len(avg)), 3.5))
    plt.bar(range(len(avg)), avg)
    plt.xlabel("component k"); plt.ylabel("mean π_k")
    plt.title("Average mixture weights π")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pi_bar.png"))
    plt.close()

    # ---------- (2) Samples from GMM -> perturbation grid ----------
    # take a single image to condition the x-dependent head
    x, _, _ = next(iter(loader))
    x = x[:1].to(device)                 # (1,C,H,W)
    if args.xdep:
        feat = encoder(x)                # (1,feat_dim)
    else:
        feat_dim = head.mu_head.in_features if head.xdep else head.mu.shape[-1]
        feat = torch.zeros(1, feat_dim, device=device)

    pi, mu, cov = head(feat)             # shapes: (1,K,*) with batch B=1
    gmm = build_gmm(pi, mu, cov, cov_type=args.cov_type)

    T = 36                                # number of samples for the grid
    z = gmm.sample((T,))                  # (T, 1, D)
    z = z.squeeze(1)                      # (T, D)

    if args.use_decoder:
        u = decoder(z)                    # (T, C, H, W)
    else:
        # direct pixel latent: D should equal C*H*W
        u = z.view(T, C, H, W)

    eps = g_ball(u, gamma=args.gamma, norm_type=args.norm)
    vutils.save_image(
        eps, os.path.join(save_dir, "samples.png"),
        nrow=int(math.sqrt(T)), normalize=True, scale_each=True
    )

    print(f"[viz] saved: {os.path.join(save_dir,'pi_bar.png')} and samples.png")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100","mnist","tinyimagenet"], default="cifar10")
    ap.add_argument("--arch", choices=["resnet18","resnet50","wide_resnet50_2","vgg16","densenet121","mobilenet_v3_large","efficientnet_b0","vit_b_16"], default="resnet18")
    ap.add_argument("--device", default="cuda:0")

    # --- NEW: external encoder controls x-dependence ---
    ap.add_argument("--encoder_backend", choices=["classifier","resnet18_imnet","vit_b_16_imnet","cnn_tiny"], default="classifier",
                    help="choose external encoder to parameterize x-dependent pi/mu/sigma")
    ap.add_argument("--encoder_ckpt", default="", help="path to your pretrained encoder (optional)")
    ap.add_argument("--freeze_encoder", action="store_true", help="freeze the external encoder")

    # Decoder control
    ap.add_argument("--use_decoder", action="store_true", default=False, help="use decoder to map latent->image noise; else direct pixel latent")
    ap.add_argument("--decoder_backend", choices=["conv","biggan256"], default="conv")
    ap.add_argument("--freeze_decoder", action="store_true")
    ap.add_argument("--gan_class", type=int, default=207)
    ap.add_argument("--gan_truncation", type=float, default=0.5)

    # GMM + training
    ap.add_argument("--num_modes", type=int, default=5)
    ap.add_argument("--latent_dim", type=int, default=64, help="latent dim (only used when --use_decoder)")
    ap.add_argument("--cov_type", choices=["diag","full","lowrank"], default="diag")
    ap.add_argument("--cov_rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=8/255)
    ap.add_argument("--mc", type=int, default=1)
    ap.add_argument("--xdep", default=True, action="store_true")
    ap.add_argument("--norm", choices=["l2","linf"], default="linf")
    ap.add_argument("--batch_size", type=int, default=64)

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset, num_classes, out_shape = get_dataset(args.dataset, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, feat_extractor = build_model(args.arch, num_classes, device)
    head, encoder, decoder = fit_phi(model, feat_extractor, loader, args, device, out_shape)
    viz_all(loader, head, encoder if args.xdep else (feat_extractor if args.encoder_backend=="classifier" else encoder),
            decoder, build_gmm, g_ball, args, out_shape, save_dir="viz")

if __name__=="__main__":
    main()
