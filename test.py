import torch
import torch.nn.functional as F
import torch.distributions as D

from irt.distributions import ReparametrizedMixtureSameFamily
from normal import StableNormal


# --------------- Lp mapping into the admissible set B -----------------

def map_to_linf_ball(eps, gamma):
    # elementwise clipping (box projection)
    return torch.clamp(eps, -gamma, gamma)

def map_to_l2_ball(eps, gamma, eps_min=1e-12):
    # project each vector in eps onto L2 ball of radius gamma
    flat = eps.view(eps.shape[:-1] + (-1,))              # [..., d]
    norms = flat.norm(p=2, dim=-1, keepdim=True).clamp_min(eps_min)
    scale = torch.minimum(torch.ones_like(norms), gamma / norms)
    flat = flat * scale
    return flat.view_as(eps)


# --------------- θ: K-component factorized Gaussian mixture ------------

class ThetaGMM(torch.nn.Module):
    """
    Factorized (diagonal) K-component mixture over R^d, with batch-size B
    (one θ per example when B == N, or shared θ when B == 1).
    Uses StableNormal + Independent + ReparametrizedMixtureSameFamily.
    """
    def __init__(self, B, K, d, init_scale=0.1, device=None):
        super().__init__()
        self.B, self.K, self.d = B, K, d
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mix_logits = torch.nn.Parameter(torch.zeros(B, K, device=device))
        self.loc        = torch.nn.Parameter(torch.zeros(B, K, d, device=device))
        # parameterize scale via softplus for positivity
        rho0 = torch.log(torch.exp(torch.tensor(init_scale, device=device)) - 1.0)
        self.rho        = torch.nn.Parameter(rho0.expand(B, K, d).clone())

    def distribution(self):
        mix = D.Categorical(logits=self.mix_logits)                # [B, K]
        scale = F.softplus(self.rho) + 1e-6                        # [B, K, d]
        base = StableNormal(loc=self.loc, scale=scale)             # univariate across d
        comp = D.Independent(base, 1)                              # event_shape=(d,)
        return ReparametrizedMixtureSameFamily(mix, comp)

    @torch.no_grad()
    def freeze_small_scales(self, min_scale=1e-5):
        # (optional) keep scales from collapsing to zero
        self.rho.data = torch.maximum(self.rho.data,
                                      torch.log(torch.exp(torch.tensor(min_scale, device=self.rho.device)) - 1.0))


# --------------- Core: learn θ to minimize WSPR (maximize loss) --------

def optimize_theta_pr_attack(
    model,
    x, y,
    gamma=0.5,
    norm="linf",          # "linf" or "l2"
    K=3,
    steps=200,
    S=64,                 # MC samples per step
    lr=5e-2,
    targeted=None,        # None for untargeted; or target class int (or tensor of shape [N])
    shared_theta=False,   # if True, share one θ across the batch
    entropy_reg=0.0,      # small positive (e.g., 1e-3) can stabilize mixture weights
    clamp01=True          # clamp images to [0,1]
):
    """
    Args:
      model: classifier returning logits.
      x: input batch [N, C, H, W]
      y: true labels [N]
      gamma, norm: perturbation budget.
      targeted: None for untargeted, or an int / LongTensor with target class(es).
      shared_theta: share θ across the whole batch (B=1) or per-example θ (B=N).
    Returns:
      theta: learned ThetaGMM
      wspr_hat: Monte-Carlo estimate of WSPR after optimization
    """
    device = next(model.parameters()).device if next(model.parameters(), None) else (x.device if x.is_cuda else 'cpu')
    model.eval()

    x = x.to(device)
    y = y.to(device)

    N = x.size(0)
    d = x[0].numel()

    B = 1 if shared_theta else N
    theta = ThetaGMM(B=B, K=K, d=d, device=device).to(device)
    opt = torch.optim.Adam(theta.parameters(), lr=lr)

    # label setup
    if targeted is None:
        # untargeted: increase CE wrt true labels (i.e., minimize correctness)
        target_labels = y
    else:
        if isinstance(targeted, int):
            target_labels = torch.full_like(y, fill_value=targeted, device=device)
        else:
            target_labels = torch.as_tensor(targeted, device=device, dtype=y.dtype)

    # helper to map eps to budget
    map_B = (map_to_linf_ball if norm == "linf" else map_to_l2_ball)

    for t in range(steps):
        opt.zero_grad(set_to_none=True)

        dist = theta.distribution()
        # Sample eps: [S, B, d]; broadcast to [S, N, d] if shared theta
        eps = dist.rsample(sample_shape=(S,))                     # [S, B, d]
        if shared_theta:
            eps = eps.expand(-1, N, -1).contiguous()              # [S, N, d]

        # map into B (L_p ball)
        eps = map_B(eps, gamma=gamma)                             # [S, N, d]

        # build perturbed inputs
        x_flat = x.view(N, -1)                                    # [N, d]
        x_pert = (x_flat.unsqueeze(0) + eps).view((S,) + x.shape) # [S, N, C, H, W]
        if clamp01:
            x_pert = torch.clamp(x_pert, 0.0, 1.0)

        # forward
        logits = model(x_pert.view(-1, *x.shape[1:]))             # [S*N, C_classes]
        if targeted is None:
            # untargeted: CE wrt true label (maximize loss)
            labels = y.repeat(S)
            loss = F.cross_entropy(logits, labels, reduction='mean')
        else:
            # targeted: CE wrt target label (minimize it to push toward target)
            labels = target_labels.repeat(S)
            loss = F.cross_entropy(logits, labels, reduction='mean')

        # entropy regularizer on mixture weights (optional)
        if entropy_reg > 0.0:
            probs = F.softmax(theta.mix_logits, dim=-1)           # [B, K]
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean()
            loss = loss - entropy_reg * ent

        # We want worst-case θ:
        # - Untargeted: maximize CE wrt true labels  -> descend on -loss
        # - Targeted:   minimize CE wrt target label -> descend on +loss
        obj = (-loss) if targeted is None else (loss)
        obj.backward()
        opt.step()

        if (t + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()
            if targeted is None:
                print(f"[{t+1:4d}/{steps}] loss={loss.item():.4f}  CE↑ acc_under_dist={acc:.3f}")
            else:
                print(f"[{t+1:4d}/{steps}] loss={loss.item():.4f}  CE↓ target_acc={acc:.3f}")

        theta.freeze_small_scales()

    # Monte-Carlo WSPR estimate after training θ
    with torch.no_grad():
        wspr_hat = estimate_wspr(model, x, y, theta, gamma=gamma, norm=norm, S=max(2048//max(1,N), 256), clamp01=clamp01)

    return theta, wspr_hat


# ---------------------- WSPR estimator --------------------------------

@torch.no_grad()
def estimate_wspr(model, x, y, theta, gamma=0.5, norm="linf", S=1024, clamp01=True):
    """
    Returns MC estimate of WSPR = P_{ε~ωθ}[ f(x+ε) = y ]
    """
    device = x.device
    N = x.size(0)
    d = x[0].numel()

    dist = theta.distribution()
    eps = dist.rsample(sample_shape=(S,))                         # [S, B, d]
    if theta.B == 1:
        eps = eps.expand(-1, N, -1).contiguous()                  # [S, N, d]
    eps = map_to_linf_ball(eps, gamma) if norm == "linf" else map_to_l2_ball(eps, gamma)

    x_flat = x.view(N, -1)
    x_pert = (x_flat.unsqueeze(0) + eps).view((S,) + x.shape)
    if clamp01:
        x_pert = torch.clamp(x_pert, 0.0, 1.0)

    logits = model(x_pert.view(-1, *x.shape[1:]))                 # [S*N, C]
    pred = logits.argmax(dim=1).view(S, N)
    corr = (pred == y.view(1, N).expand(S, N)).float().mean().item()
    return corr


# ---------------------- Example usage ---------------------------------
if __name__ == "__main__":
    # toy example (replace with your real model/x/y)
    class TinyNet(torch.nn.Module):
        def __init__(self, C=1, num_classes=10):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(C*28*28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes),
            )
        def forward(self, z): return self.net(z)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyNet(C=1).to(device).eval()

    # one MNIST-like batch
    N = 8
    x = torch.rand(N, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (N,), device=device)

    theta, wspr = optimize_theta_pr_attack(
        model, x, y,
        gamma=8/255,          # typical L_inf budget for images
        norm="linf",
        K=3,
        steps=100,
        S=32,
        lr=5e-2,
        targeted=None,        # e.g., 2 for targeted-to-class-2
        shared_theta=False,   # True = one θ for all N; False = per-example θ
        entropy_reg=1e-3
    )
    print("Estimated WSPR after optimization:", wspr)
