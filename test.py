import torch
import torch.nn as nn
from pl_bolts.models.autoencoders import AE
from fit_classifiers import get_dataset

class WarpDecoder(nn.Module):
    """
    Wrapper for decoders that take a flat latent z of shape [B, D].

    - Enforces 2D latent inputs [batch, dim].
    - Optionally auto-projects mismatched latent size to the decoder's expected size
      using a single Linear layer (disabled by default for safety).
    """

    def __init__(self, decoder: nn.Module, latent_dim: int = None):
        """
        Args:
            decoder: The underlying decoder module. Ideally its first layer is a Linear
                     with attribute `in_features` (e.g., decoder.linear.in_features).
            latent_dim: Expected latent size if it cannot be inferred from the decoder.
        """

        super().__init__()
        self.decoder = decoder

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


# Build datasets/loaders
test_set, _ = get_dataset('cifar10', "./dataset", False, 32, True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=8, shuffle=False,
    num_workers=4, pin_memory=True
)

for x, y in test_loader:
    print(x.shape, y.shape)

    ae = AE(input_height=32)
    # Example 1: pl_bolts AE where decoder starts with Linear(256 -> ...).
    # The wrapper will infer target_dim=256 from decoder.linear.in_features.
    warp = WarpDecoder(ae.decoder, latent_dim=256)            # strict (no auto projection)
    z = ae.encode(x)                      # or: ae.fc(ae.encoder(x).flatten(1))
    x_prim = warp(z)

