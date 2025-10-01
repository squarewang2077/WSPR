import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers import AutoencoderKL
from pl_bolts.models.autoencoders import AE


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


### -------------------- External Decoder -------------------- ###

class WarpDecoder(nn.Module):
    """
    Wrapper for decoders that take a flat latent z of shape [B, D].

    - Enforces 2D latent inputs [batch, dim].
    - Optionally auto-projects mismatched latent size to the decoder's expected size
      using a single Linear layer (disabled by default for safety).
    """

    def __init__(self, decoder: nn.Module, latent_dim: int = None, device=None):
        """
        Args:
            decoder: The underlying decoder module. Ideally its first layer is a Linear
                     with attribute `in_features` (e.g., decoder.linear.in_features).
            latent_dim: Expected latent size if it cannot be inferred from the decoder.
        """

        super().__init__()
        self.decoder = decoder.to(device) if device is not None else decoder

        # Try to infer the expected input dimension from a common pattern:
        # a `linear` submodule with `in_features`. If not available, require latent_dim.
        inferred_lat_dim = getattr(getattr(decoder, "linear", None), "in_features", None)
        if latent_dim is None:
            raise ValueError(
                "Pass latent_dim explicitly (e.g., latent_dim=256) "
            )
        elif inferred_lat_dim is None:
            raise ValueError(
                "Cannot infer decoder's expected latent size. "
            )
        elif latent_dim != inferred_lat_dim:
            raise ValueError(
                "The decoder's expected latent size is mismatched: "
            )
        else:
            self.latent_dim = latent_dim


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape [B, D].

        Returns:
            Decoded tensor produced by `self.decoder(z_projected)`.
        """
        # check whether z is 2D and floating point
        if z.dim() != 2:
            raise ValueError(f"`z` must be 2D [B, D], got shape {tuple(z.shape)}")
        if not z.is_floating_point():
            z = z.float()

        _, D = z.shape

        # check latent dim of input z
        if D != self.latent_dim:
            raise ValueError(
                f"Latent size mismatch: got {D}, expected {self.latent_dim}. "
            )

        return self.decoder(z)




## the function to load different decoders
def load_decoder_backend(backend, latent_dim, out_shape, device, freeze,
                         gan_class=207, gan_trunc=0.5,
                         pca_path=None, dwt_levels=2, dwt_wavelet="haar"
                        ):
    # unify the backend string
    backend = backend.lower()
    dec = None
    dec_latent = None

    # the decoder of AE
    if backend == "ae":

        ae = AE(input_height=32)
        ae = ae.from_pretrained('cifar10-resnet18')
        ae.freeze()

        dec = WarpDecoder(ae.decoder, latent_dim=latent_dim, device=device)
        dec_latent = latent_dim

    else:
        raise ValueError(f"unknown decoder_backend: {backend}")

    return dec, dec_latent