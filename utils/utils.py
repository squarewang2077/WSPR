import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from attacks import WithIndex
import torch.nn.functional as F
from fit_classifiers import build_model as build_clf_model


### -------------------- Dataset -------------------- ###
def get_dataset(name, root="./dataset", train=False, resize=False):
    """
    Get a dataset by name.
    Returns (dataset, num_classes, input_shape)
    """

    # Default values
    num_classes = None
    input_shape = None

    name = name.lower()
    if name == "cifar10":
        mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
        
        if resize:
            tf = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean,std)])
            input_shape = (3,224,224)
        else:
            tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
            input_shape = (3,32,32)
    
        num_classes = 10
        ds = WithIndex(torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf))    

    elif name == "cifar100":
        mean, std = (0.5071,0.4865,0.4409), (0.2673,0.2564,0.2762)
        if resize:
            tf = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean,std)])
            input_shape = (3,224,224)
        else:
            tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
            input_shape = (3,32,32)

        num_classes = 100
        ds = WithIndex(torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=tf))

    elif name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        if resize:
            tf = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean,std)])
            input_shape = (1,224,224)
        else:
            tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
            input_shape = (1,28,28)

        num_classes = 10
        ds = WithIndex(torchvision.datasets.MNIST(root=root, train=train, download=True, transform=tf))

    elif name == "tinyimagenet":
        mean, std = (0.4802,0.4481,0.3975), (0.2302,0.2265,0.2262)
        if resize:
            tf = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean,std)])
            input_shape = (3,224,224)
        else:
            tf = T.Compose([T.Resize(64),T.CenterCrop(64),T.ToTensor(),T.Normalize(mean,std)])
            input_shape = (3,64,64)

        num_classes = 200
        ds = WithIndex(torchvision.datasets.ImageFolder(os.path.join(root,"tiny-imagenet-200","val"), transform=tf))

    else: # raise error if unknown dataset
        raise ValueError(f"Unknown dataset {name}")

    return ds, num_classes, input_shape



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


### -------------------- Classifier factory -------------------- ###
def build_model(arch: str, num_classes: int, device):

    arch = arch.lower()
    if arch == "resnet18": # this defaults to training from scratch
        # There is a version of resize pretrained resnet18, but not considered here 
        # it is from an old plan    
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    # from here all model are pretrained and finetuned
    elif arch == "resnet50":
        model = build_clf_model(arch="resnet50", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    elif arch == "wide_resnet50_2":
        model = build_clf_model(arch="wide_resnet50_2", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    elif arch == "vgg16":
        model = build_clf_model(arch="vgg16", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())

    elif arch == "densenet121":
        model = build_clf_model(arch="densenet121", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(model.features, nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    elif arch == "mobilenet_v3_large":
        model = build_clf_model(arch="mobilenet_v3_large", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    elif arch == "efficientnet_b0":
        model = build_clf_model(arch="efficientnet_b0", num_classes=num_classes, device=device, pretrained=False)
        feat_extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    elif arch == "vit_b_16":
        model = build_clf_model(arch="vit_b_16", num_classes=num_classes, device=device, pretrained=False)
        class ViTFeat(nn.Module):
            def __init__(self, vit): 
                super().__init__(); self.vit = vit
            
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
    # model = model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()] 
    # feat_extractor = feat_extractor.to(device).eval(); [p.requires_grad_(False) for p in feat_extractor.parameters()]
    
    return model, feat_extractor


@torch.no_grad()
def infer_feat_dim(fe: nn.Module, img_shape):
    '''
        Return the feature dimension for classifier
    '''
    C,H,W = img_shape
    dummy = torch.zeros(1, C, H, W, device=next(fe.parameters()).device)

    return fe(dummy).shape[-1]


### Evaluation ### 
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



### -------------------- g_B -------------------- ###
def g_ball(u, gamma, norm_type):
    '''
        Mapping to the perturbation budget
    '''
    g = None 

    if norm_type == "linf":
        g = gamma * u.tanh() # using tanh for L-infty

    elif norm_type == "l2": # project onto l2 ball instead of tanh for stability

        flat = u.view(u.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        g = (gamma * flat / norm).view_as(u)

    if g is None:
        raise ValueError(f"not supported norm_type: {norm_type}")

    return g
    

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



@torch.no_grad()
def compute_pr_on_clean_correct_old(
    model, head, encoder, loader, args, out_shape,
    decoder=None, S=100, batch_indices=None
):
    """
    PR on clean-correct set:
      PR = E_{(x,y) ∈ CleanCorrect} E_{eps~GMM_phi(.|x)} [ 1{ f(x+g_B(eps)) = y } ]

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
        x, y = x.to(device), y.to(device)
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
            feat = torch.empty(n, 0, device=device)

        # only batch size matters when xdep=False
        pi, mu, cov = head(feat)  # shapes (n,K,*)
        gmm = head.mixture(pi, mu, cov, cov_type=args.cov_type)

        # sample S per selected item → [S, n, D]
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

    print(f"[PR@clean] used={total_used} / seen={total_seen} "
          f"(clean acc={clean_acc*100:.2f}%), S={S} → PR={pr:.4f}")

    return pr, total_used, clean_acc


### Additional utility functions for the class of gmm4pr which is a advanced version ###
def slug_gamma(g):
    """Make gamma filename-safe."""
    return f"{g:.4f}".replace('.', 'p')


def initialize_gmm_parameters(gmm, init_mode='spread'): 
    with torch.no_grad():
        if hasattr(gmm, 'mu') and isinstance(gmm.mu, nn.Parameter):
            K, D = gmm.mu.shape
            
            if init_mode == 'spread':
                # Original binary pattern
                gmm.mu.data.normal_(0, 0.5)
                if K <= 8 and D >= 3:
                    for k in range(min(K, 8)):
                        binary = format(k, '03b')
                        for d, bit in enumerate(binary):
                            if d < D:
                                gmm.mu.data[k, d] = 1.0 if bit == '1' else -1.0
            
            elif init_mode == 'random':
                gmm.mu.data.normal_(0, 1.0)
            
            elif init_mode == 'grid':
                # Evenly spaced grid
                if D >= 2:
                    side = int(np.ceil(K ** (1/2)))
                    for k in range(K):
                        i, j = k // side, k % side
                        gmm.mu.data[k, 0] = (i / side) * 2 - 1
                        gmm.mu.data[k, 1] = (j / side) * 2 - 1
            
            elif init_mode == 'uniform':
                # Uniform in [-1, 1]
                gmm.mu.data.uniform_(-1, 1)
    
    print(f"[init] GMM means initialized with mode='{init_mode}'")

class TemperatureScheduler:
    """Temperature scheduler."""
    def __init__(self, gmm, initial_T_pi=2.0, final_T_pi=1.0, 
                 initial_T_sigma=1.5, final_T_sigma=1.0,
                 initial_T_shared=1.0, final_T_shared=1.0,
                 warmup_epochs=50):
        self.gmm = gmm
        self.initial_T_pi = initial_T_pi
        self.final_T_pi = final_T_pi
        self.initial_T_sigma = initial_T_sigma
        self.final_T_sigma = final_T_sigma
        self.initial_T_shared = initial_T_shared
        self.final_T_shared = final_T_shared
        self.warmup_epochs = warmup_epochs
        
    def step(self, epoch):
        """Update temperatures."""
        if epoch <= self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            T_pi = self.initial_T_pi + alpha * (self.final_T_pi - self.initial_T_pi)
            T_sigma = self.initial_T_sigma + alpha * (self.final_T_sigma - self.initial_T_sigma)
            T_shared = self.initial_T_shared + alpha * (self.final_T_shared - self.initial_T_shared)
        else:
            T_pi = self.final_T_pi
            T_sigma = self.final_T_sigma
            T_shared = self.final_T_shared
        
        self.gmm.set_temperatures(T_pi=T_pi, T_sigma=T_sigma, T_shared=T_shared)
        return T_pi, T_sigma, T_shared


@torch.no_grad()
def check_mode_collapse(gmm, loader, device, num_batches=10):
    """Check mode collapse."""
    gmm.eval()
    pi_distributions = []
    
    for i, (x, y, _) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        
        out = gmm.forward(x=x, y=y)
        pi_logits = out['cache']['pi_logits']
        pi_probs = F.softmax(pi_logits, dim=-1)
        pi_distributions.append(pi_probs.cpu())
    
    all_pi = torch.cat(pi_distributions, dim=0)
    mean_pi = all_pi.mean(dim=0)
    max_pi = mean_pi.max().item()
    min_pi = mean_pi.min().item()
    std_pi = mean_pi.std().item()
    entropy = -(mean_pi * torch.log(mean_pi + 1e-8)).sum().item()
    max_entropy = np.log(gmm.K)
    
    print(f"\n{'='*60}")
    print(f"MODE COLLAPSE CHECK (K={gmm.K})")
    print(f"{'='*60}")
    print(f"Average π per component: {mean_pi.numpy()}")
    print(f"Max π: {max_pi:.4f} | Min π: {min_pi:.4f} | Std: {std_pi:.4f}")
    print(f"Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")
    
    if max_pi > 0.5:
        print(f"⚠️  WARNING: Potential mode collapse!")
    elif std_pi > 0.15:
        print(f"⚠️  WARNING: High variance in usage")
    else:
        print(f"✓ Component usage looks balanced")
    print(f"{'='*60}\n")
    
    gmm.train()
    return {
        'mean_pi': mean_pi.numpy(),
        'max_pi': max_pi,
        'min_pi': min_pi,
        'std_pi': std_pi,
        'entropy': entropy,
        'entropy_ratio': entropy / max_entropy
    }


def build_decoder_from_flag(backend: str, latent_dim: int, out_shape: tuple, device):
    """
    Build decoder that maps latent_dim -> out_shape.
    
    Args:
        backend: Decoder type ('bicubic', 'wavelet', 'dct', 'nearest_blur', 
                                'conv', 'upsample', 'tiny', 'mlp')
        latent_dim: Dimensionality of latent space
        out_shape: Output shape (C, H, W)
        device: Target device
    
    Returns:
        decoder: nn.Module that maps [B, latent_dim] -> [B, C, H, W]
    """
    C, H, W = out_shape
    
    # Helper function to calculate adaptive sizes
    def calc_init_size(target_size):
        """Calculate initial spatial size for progressive upsampling."""
        # Start from 4x4 or 7x7, whichever is more appropriate
        if target_size <= 32:
            return 4  # For CIFAR-10, MNIST, etc.
        elif target_size <= 64:
            return 7  # For 64x64 images
        else:
            return target_size // 32  # For larger images
    
    def calc_num_upsample_layers(init_size, target_size):
        """Calculate number of 2x upsampling layers needed."""
        num_layers = 0
        current = init_size
        while current < target_size:
            current *= 2
            num_layers += 1
        return num_layers
    
    # ============ FROZEN DECODERS ============
    if backend == "bicubic":
        class BicubicDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.init_size = calc_init_size(min(H, W))
                init_dim = C * self.init_size * self.init_size
                self.latent_to_spatial = nn.Linear(latent_dim, init_dim)
            
            def forward(self, z):
                B = z.size(0)
                h = self.latent_to_spatial(z)
                h = h.view(B, C, self.init_size, self.init_size)
                return F.interpolate(h, size=(H, W), mode='bicubic', align_corners=False)
        
        decoder = BicubicDecoder().to(device)
        print(f"[Decoder 'bicubic'] {sum(p.numel() for p in decoder.parameters()):,} params (frozen decoder)")
    
    elif backend == "wavelet":
        try:
            import pywt
            class SimpleWaveletDecoder(nn.Module):
                def __init__(self, wavelet='haar'):
                    super().__init__()
                    self.wavelet = wavelet
                    self.level = 2
                    # Calculate size after wavelet decomposition
                    h, w = H, W
                    for _ in range(self.level):
                        h = (h + 1) // 2
                        w = (w + 1) // 2
                    coeff_size = h * w * 4  # 4 subbands per level
                    self.latent_to_coeffs = nn.Linear(latent_dim, coeff_size * C)
                    self.coeff_h, self.coeff_w = h, w
                
                def forward(self, z):
                    B = z.size(0)
                    coeffs = self.latent_to_coeffs(z).view(B, C, self.coeff_h, self.coeff_w)
                    # Use bilinear interpolation as approximation
                    return F.interpolate(coeffs, size=(H, W), mode='bilinear', align_corners=False)
            
            decoder = SimpleWaveletDecoder().to(device)
            print(f"[Decoder 'wavelet'] {sum(p.numel() for p in decoder.parameters()):,} params (frozen decoder)")
        except ImportError:
            print("Warning: pywt not found, falling back to bicubic")
            return build_decoder_from_flag("bicubic", latent_dim, out_shape, device)
    
    elif backend == "dct":
        class DCTDecoder(nn.Module):
            def __init__(self, compression=4):
                super().__init__()
                self.compression = compression
                dct_h, dct_w = max(1, H // compression), max(1, W // compression)
                dct_size = C * dct_h * dct_w
                self.latent_to_dct = nn.Linear(latent_dim, dct_size)
                self.dct_h, self.dct_w = dct_h, dct_w
            
            def forward(self, z):
                B = z.size(0)
                dct = self.latent_to_dct(z).view(B, C, self.dct_h, self.dct_w)
                return F.interpolate(dct, size=(H, W), mode='bilinear', align_corners=False)
        
        decoder = DCTDecoder().to(device)
        print(f"[Decoder 'dct'] {sum(p.numel() for p in decoder.parameters()):,} params (frozen decoder)")
    
    elif backend == "nearest_blur":
        class NearestBlurDecoder(nn.Module):
            def __init__(self, blur_kernel=5):
                super().__init__()
                self.init_size = calc_init_size(min(H, W))
                init_dim = C * self.init_size * self.init_size
                self.latent_to_spatial = nn.Linear(latent_dim, init_dim)
                
                # Create Gaussian blur kernel
                sigma = blur_kernel / 6.0
                x = torch.arange(blur_kernel).float() - blur_kernel // 2
                gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
                kernel_1d = gauss / gauss.sum()
                kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
                kernel = kernel_2d.expand(C, 1, blur_kernel, blur_kernel).contiguous()
                self.register_buffer('blur_kernel', kernel)
            
            def forward(self, z):
                B = z.size(0)
                h = self.latent_to_spatial(z).view(B, C, self.init_size, self.init_size)
                h = F.interpolate(h, size=(H, W), mode='nearest')
                padding = self.blur_kernel.size(-1) // 2
                return F.conv2d(h, self.blur_kernel, padding=padding, groups=C)
        
        decoder = NearestBlurDecoder().to(device)
        print(f"[Decoder 'nearest_blur'] {sum(p.numel() for p in decoder.parameters()):,} params (frozen decoder)")
    
    # ============ TRAINABLE DECODERS ============
    elif backend in ["conv", "convtranspose"]:
        class ConvDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Adaptive architecture based on target size
                target_size = max(H, W)
                self.init_size = calc_init_size(target_size)
                num_layers = calc_num_upsample_layers(self.init_size, target_size)
                
                # Initial projection
                self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
                
                # Build upsampling layers dynamically
                layers = []
                in_ch = 256
                for i in range(num_layers):
                    out_ch = max(16, in_ch // 2)  # Halve channels, min 16
                    layers.extend([
                        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                    ])
                    in_ch = out_ch
                
                # Final layer to match output channels
                layers.append(nn.Conv2d(in_ch, C, 3, 1, 1))
                self.decoder = nn.Sequential(*layers)
            
            def forward(self, z):
                B = z.size(0)
                h = self.fc(z).view(B, 256, self.init_size, self.init_size)
                out = self.decoder(h)
                # Adjust to exact target size if needed
                if out.shape[2:] != (H, W):
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                return out
        
        decoder = ConvDecoder().to(device)
        print(f"[Decoder 'conv'] {sum(p.numel() for p in decoder.parameters()):,} params (trainable)")
    
    elif backend == "upsample":
        class UpsampleDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                target_size = max(H, W)
                self.init_size = calc_init_size(target_size)
                num_layers = calc_num_upsample_layers(self.init_size, target_size)
                
                self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
                
                # Build upsampling layers
                layers = []
                in_ch = 256
                for i in range(num_layers):
                    out_ch = max(16, in_ch // 2)
                    layers.extend([
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True),
                    ])
                    in_ch = out_ch
                
                layers.append(nn.Conv2d(in_ch, C, 3, 1, 1))
                self.decoder = nn.Sequential(*layers)
            
            def forward(self, z):
                B = z.size(0)
                h = self.fc(z).view(B, 256, self.init_size, self.init_size)
                out = self.decoder(h)
                if out.shape[2:] != (H, W):
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                return out
        
        decoder = UpsampleDecoder().to(device)
        print(f"[Decoder 'upsample'] {sum(p.numel() for p in decoder.parameters()):,} params (trainable)")
    
    elif backend == "tiny":
        class TinyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                target_size = max(H, W)
                self.init_size = calc_init_size(target_size)
                num_layers = calc_num_upsample_layers(self.init_size, target_size)
                
                self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
                
                # Minimal architecture
                layers = []
                in_ch = 128
                for i in range(num_layers):
                    out_ch = max(8, in_ch // 2)
                    layers.extend([
                        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                        nn.ReLU(True),
                    ])
                    in_ch = out_ch
                
                layers.append(nn.Conv2d(in_ch, C, 3, 1, 1))
                self.decoder = nn.Sequential(*layers)
            
            def forward(self, z):
                B = z.size(0)
                h = self.fc(z).view(B, 128, self.init_size, self.init_size)
                out = self.decoder(h)
                if out.shape[2:] != (H, W):
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                return out
        
        decoder = TinyDecoder().to(device)
        print(f"[Decoder 'tiny'] {sum(p.numel() for p in decoder.parameters()):,} params (trainable)")
    
    elif backend in ["mlp", "linear"]:
        out_dim = C * H * W
        class LinearDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Adaptive hidden sizes based on output size
                hidden1 = max(512, latent_dim * 4)
                hidden2 = min(2048, out_dim // 2)
                
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden1),
                    nn.ReLU(True),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(True),
                    nn.Linear(hidden2, out_dim),
                )
            
            def forward(self, z):
                return self.net(z).view(z.size(0), C, H, W)
        
        decoder = LinearDecoder().to(device)
        print(f"[Decoder 'mlp'] {sum(p.numel() for p in decoder.parameters()):,} params (trainable)")
    
    else:
        raise ValueError(
            f"Unknown decoder backend: '{backend}'. Choose from:\n"
            f"  Frozen:    bicubic, wavelet, dct, nearest_blur\n"
            f"  Trainable: conv, upsample, tiny, mlp"
        )
    
    return decoder