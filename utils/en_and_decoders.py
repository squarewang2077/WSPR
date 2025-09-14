import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

### -------------------- External Encoder -------------------- ###
class TinyCNNEncoder(nn.Module): # NOT ACTUALLY USED
    """A trainable encoder (not relying on ImageNet pre-training)"""

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
    return: encoder(nn.Module), feat_dim(int)
    backend:
      - 'classifier': reuse feat_extractor
      - 'resnet18_imnet': torchvision resnet18 (ImageNet pretrained) remove fc
      - 'vit_b_16_imnet': torchvision vit_b_16 (ImageNet pretrained) taking CLS
      - 'cnn_tiny': lightweight custom CNN encoder (trainable)
    """
    backend = backend.lower()

    if backend == "classifier": # reuse features from classifiers
        
        return None, None

    if backend == "resnet18_imnet":
        try:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            pretrained_used = True

        except Exception:
            model = torchvision.models.resnet18(weights=None)
            pretrained_used = False

        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten()) # remove the final fc layer
        feat_dim = 512

    elif backend == "vit_b_16_imnet":
        try:
            vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
            pretrained_used = True
        except Exception:
            vit = torchvision.models.vit_b_16(weights=None)
            pretrained_used = False

        class ViTFeat(nn.Module): # to extract the cls token
            def __init__(self, m): 
                super().__init__()
                self.m=m 
            def forward(self,x):
                x=self.m._process_input(x) # patch embedding 
                n=x.shape[0]
                cls=self.m.class_token.expand(n,-1,-1) # cls_token is of [1, 1, D]
                x=torch.cat([cls,x],dim=1)
                x=self.m.encoder(x) # transformer encoder
                x=x[:,0]
                return self.m.ln(x) if hasattr(self.m,"ln") else x # if has layernorm 

        encoder = ViTFeat(vit)
        feat_dim = vit.hidden_dim

    elif backend == "cnn_tiny": # NOT supposed to be used! since It has to be trained!
        encoder = TinyCNNEncoder(in_shape=img_shape, out_dim=256)
        feat_dim = 256
        pretrained_used = False

    else:
        raise ValueError("encoder_backend must be one of {'classifier','resnet18_imnet','vit_b_16_imnet','cnn_tiny'}")

    print(f"[debug] encoder backend={backend}, pretrained={pretrained_used}") # check whether the encoder is pretrained!

    encoder = encoder.to(device)

    # download the encoder ckpt if provided
    if ckpt and os.path.isfile(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
            state = {k.replace("module.",""): v for k,v in state.items()}
            encoder.load_state_dict(state, strict=False)
            print("[info] external encoder ckpt loaded.")
        except Exception as e:
            print(f"[warn] failed to load encoder ckpt: {e}")

    if freeze: # whether to freeze the encoder
        for p in encoder.parameters(): p.requires_grad_(False)
        encoder.eval()
        print("[info] external encoder frozen.")
    else:
        print("[info] external encoder will be trained.")


    return encoder, feat_dim

# ====== 1) PCA 固定解码器 ======
class PCADecoder(nn.Module):
    """
    线性解码：x ≈ U z + mean
    其中 U: [P, D]，mean: [P]，P = C*H*W
    """
    def __init__(self, U: torch.Tensor, mean: torch.Tensor, out_shape):
        super().__init__()
        assert U.dim()==2 and mean.dim()==1, "U [P,D], mean [P]"
        self.register_buffer("U", U)         # 不训练
        self.register_buffer("mean", mean)   # 不训练
        self.out_shape = out_shape           # (C,H,W)

    def forward(self, z):                    # z: [B, D]
        x_flat = z @ self.U.t() + self.mean  # [B, P]
        return x_flat.view(z.size(0), *self.out_shape)


# ====== 2) Haar 小波解码器（只用低频系数） ======
class HaarDWTDecoder(nn.Module):
    """
    只用 level 级的 cA 低频系数作为潜变量，细节系数置 0 后逆变换。
    依赖 pywt（pip install pywavelets）。若未安装会抛错。
    D = C * (H/2^L) * (W/2^L)
    """
    def __init__(self, out_shape=(3,32,32), levels=2, wavelet="haar"):
        super().__init__()
        try:
            import pywt  # noqa: F401
        except Exception as e:
            raise RuntimeError("HaarDWTDecoder 需要安装 pywavelets: pip install pywavelets") from e
        C,H,W = out_shape
        assert H % (2**levels)==0 and W % (2**levels)==0, "H,W 必须能被 2^levels 整除"
        self.out_shape = out_shape
        self.levels = levels
        self.wavelet = wavelet
        self.cAh = H // (2**levels)
        self.cAw = W // (2**levels)
        self.latent_dim = C * self.cAh * self.cAw

    def forward(self, z):  # z: [B, D]
        import pywt
        B = z.size(0)
        C,H,W = self.out_shape
        device = z.device
        # reshape 成每通道的 cA 低频系数
        z = z.view(B, C, self.cAh, self.cAw).detach().cpu().numpy()
        out = []
        for b in range(B):
            chans = []
            for c in range(C):
                cA = z[b, c]
                coeffs = cA
                # 逆分解需要 (cA, (cH,cV,cD))*levels 结构；细节系数全 0
                for _ in range(self.levels):
                    zeros = (np.zeros_like(cA), np.zeros_like(cA), np.zeros_like(cA))
                    coeffs = (coeffs, zeros)
                img = pywt.waverec2(coeffs, wavelet=self.wavelet)
                chans.append(torch.from_numpy(img))
            img = torch.stack(chans, dim=0)  # [C,H,W]
            out.append(img)
        out = torch.stack(out, dim=0).to(device)  # [B,C,H,W]
        return out


# ====== 3) 预训练 VAE 解码器（以 SD VAE 为例） ======
class SDVAEDecoder(nn.Module):
    """
    使用 diffusers 的 AutoencoderKL 作为固定 decoder。
    约定：H,W 与 VAE 下采样比例一致（一般 /8），latent 维度 = 4*(H/8)*(W/8)
    """
    def __init__(self, out_shape=(3,256,256), repo_id="stabilityai/sd-vae-ft-mse", vae_scale=0.18215, dtype=torch.float16, device="cuda"):
        super().__init__()
        try:
            from diffusers import AutoencoderKL # package from huggingface
        except Exception as e:
            raise RuntimeError("需要安装 diffusers：pip install diffusers accelerate transformers safetensors") from e
        self.out_shape = out_shape
        self.vae = AutoencoderKL.from_pretrained(repo_id).to(device, dtype=dtype).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae_scale = vae_scale
        C,H,W = out_shape
        self.zc = 4
        self.zh = H // 8
        self.zw = W // 8
        self.latent_dim = self.zc * self.zh * self.zw

    @torch.no_grad()
    def forward(self, z):  # z: [B, D]
        B = z.size(0)
        z = z.view(B, self.zc, self.zh, self.zw).to(self.vae.device, dtype=self.vae.dtype)
        z = z * self.vae_scale
        x = self.vae.decode(z).sample  # [B,3,H,W], 范围通常在 [-1,1]
        return x.to(dtype=torch.float32)



def load_decoder_backend(backend, latent_dim, out_shape, device, freeze,
                         gan_class=207, gan_trunc=0.5,
                         pca_path=None, dwt_levels=2, dwt_wavelet="haar",
                         sd_vae_repo="stabilityai/sd-vae-ft-mse", sd_vae_dtype="fp16"):
    backend = backend.lower()

    if backend == "pca_fixed":
        assert pca_path is not None and os.path.isfile(pca_path), "--pca_path 未提供或不存在"
        chk = torch.load(pca_path, map_location="cpu")
        U, mean = chk["U"], chk["mean"]  # U:[P,D], mean:[P]
        dec = PCADecoder(U.to(device), mean.to(device), out_shape).to(device)
        [p.requires_grad_(False) for p in dec.parameters()]
        return dec, U.shape[1]

    elif backend == "haar_dwt":
        dec = HaarDWTDecoder(out_shape=out_shape, levels=dwt_levels, wavelet=dwt_wavelet).to(device)
        [p.requires_grad_(False) for p in dec.parameters()]
        return dec, dec.latent_dim

    elif backend == "sd_vae":
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[sd_vae_dtype]
        dec = SDVAEDecoder(out_shape=out_shape, repo_id=sd_vae_repo, dtype=dtype, device=device).to(device)
        [p.requires_grad_(False) for p in dec.parameters()]
        return dec, dec.latent_dim

    else:
        raise ValueError(f"unknown decoder_backend: {backend}")
