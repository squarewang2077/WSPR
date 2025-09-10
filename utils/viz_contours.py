import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def _kde2d(points_2d: torch.Tensor,
           xmin: float, xmax: float, ymin: float, ymax: float,
           gridsize: int = 200, bandwidth: float = None):
    """
    Minimal 2D KDE on a uniform grid using an isotropic Gaussian kernel.

    points_2d: (N,2) tensor on CUDA/CPU
    returns: (X, Y, Z) where X,Y are (G,G) meshgrid (numpy), Z is (G,G) density (numpy)
    """
    device = points_2d.device
    N = points_2d.size(0)
    if N == 0:
        raise ValueError("kde2d: got empty points.")

    # Scott's rule for bandwidth if not given
    if bandwidth is None:
        std = points_2d.std(dim=0)  # (2,)
        std = torch.clamp(std, min=1e-6)  # avoid zero bandwidth
        h = 1.06 * std.mean() * (float(N) ** (-1.0 / 5.0))
    else:
        h = torch.tensor(float(bandwidth), device=device)

    # grid
    xs = torch.linspace(xmin, xmax, gridsize, device=device)
    ys = torch.linspace(ymin, ymax, gridsize, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")     # (G,G)

    # compute Gaussian density: sum_i N((x,y) | point_i, h^2 I)
    # Z[gx,gy] = mean_i exp(-||grid - p_i||^2 / (2 h^2))
    # Vectorized computation
    Px = points_2d[:, 0].view(-1, 1, 1)              # (N,1,1)
    Py = points_2d[:, 1].view(-1, 1, 1)              # (N,1,1)
    dx2 = (X.unsqueeze(0) - Px) ** 2                 # (N,G,G)
    dy2 = (Y.unsqueeze(0) - Py) ** 2
    dist2 = dx2 + dy2
    Z = torch.exp(- dist2 / (2.0 * h * h)).mean(dim=0)  # (G,G), average to form a density-like map
    Z = Z / (Z.max().clamp_min(1e-12))  # normalize to [0,1] for nicer contours

    # move to numpy for matplotlib
    Xn, Yn, Zn = X.detach().cpu().numpy(), Y.detach().cpu().numpy(), Z.detach().cpu().numpy()
    return Xn, Yn, Zn


def _contour_on_ax(ax, samples_2d: torch.Tensor, title: str,
                   gridsize: int = 200, pad_scale: float = 0.1):
    """
    Helper: draw filled contour + contour lines from 2D samples onto ax.
    """
    if samples_2d.numel() == 0:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return

    # pad ranges a bit to avoid boundary clipping
    x = samples_2d[:, 0]; y = samples_2d[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    rx, ry = xmax - xmin, ymax - ymin
    if rx <= 0: rx = 1.0
    if ry <= 0: ry = 1.0
    xmin -= pad_scale * rx; xmax += pad_scale * rx
    ymin -= pad_scale * ry; ymax += pad_scale * ry

    X, Y, Z = _kde2d(samples_2d, xmin, xmax, ymin, ymax, gridsize=gridsize)

    ax.contourf(X, Y, Z, levels=12, alpha=0.85)   # filled contours
    ax.contour(X, Y, Z, levels=12, linewidths=0.8)     # contour lines
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def _latent_to_eps(z: torch.Tensor, use_decoder: bool, decoder,
                   C: int, H: int, W: int,
                   gamma: float, norm_type: str,
                   g_ball_fn=None):
    """
    Map latent z -> u (decoder or reshape) -> eps = g_B(u).
    z: (N, D)
    returns eps_flat: (N, C*H*W)
    """
    N, D = z.shape
    if use_decoder:
        u = decoder(z)                      # typically (N, C, H, W); or (N, D) depending on your decoder
        if u.dim() == 2:                    # in case decoder returns (N,D), try reshape later
            u = u.view(N, C, H, W)
    else:
        u = z.view(N, C, H, W)

    if g_ball_fn is None:
        raise ValueError("g_ball_fn must be provided (your projection g_B).")
    eps = g_ball_fn(u, gamma=gamma, norm_type=norm_type)
    return eps.view(N, -1)                  # flatten for PCA/KDE


def _project_pca_2d(X: torch.Tensor):
    """
    Generic PCA->2D for samples: X (N,D) -> (mean (D,), P(2,D), Y(N,2)).
    Uses SVD; if D<2 or rank<2, we pad safely.
    """
    N, D = X.shape
    Xm = X.mean(0, keepdim=True)
    Xc = X - Xm
    if D >= 2:
        try:
            # full_matrices=False -> economy SVD
            U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
            P = Vt[:2, :]                   # (2,D)
            Y = Xc @ P.T                    # (N,2)
        except RuntimeError:
            P = torch.zeros(2, D, device=X.device); P[0, 0] = 1.0
            Y = torch.cat([Xc[:, :1], torch.zeros(N, 1, device=X.device)], dim=1)
    else:
        P = torch.zeros(2, D, device=X.device); P[0, 0] = 1.0
        Y = torch.cat([Xc, torch.zeros(N, 1, device=X.device)], dim=1)
    return Xm.squeeze(0), P, Y


@torch.no_grad()
def viz_pca_contours(loader,
                     head,                # GMM or your GMMHead/GMM class (implements forward(feat))
                     encoder,             # encoder module or None (when xdep=False)
                     decoder,             # decoder module or None
                     build_gmm_fn,        # function to build torch.distributions mixture (build_gmm)
                     g_ball_fn,           # g_B function (gamma/norm_type projection)
                     args,                # to read args.xdep / args.cov_type / args.gamma / args.norm etc.
                     out_shape,           # (C,H,W)
                     save_dir="viz",
                     n_inputs_when_xdep=8,
                     S_latent=5000,
                     gridsize=200):
    """
    Draw contour maps for:
      (A) latent distribution (z) after PCA->2D;
      (B) distribution after g_B: z -> u -> eps, then PCA->2D of eps, then contour.

    - If x-independent (args.xdep=False): draw a single pair of figures (latent + eps).
    - If x-dependent (args.xdep=True): pick first `n_inputs_when_xdep` images from loader
      and draw a grid of subplots (each input has 1 latent contour + 1 eps contour in separate figures).

    Saved files:
      - 'contour_latent_xindep.png' / 'contour_eps_xindep.png'
      - 'contour_latent_xdep.png'   / 'contour_eps_xdep.png'
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(head.parameters()).device if hasattr(head, "parameters") else torch.device("cpu")
    C, H, W = out_shape

    # -------- Case 1: x-independent --------
    if not args.xdep:
        # create a dummy "batch" so head can expand params with correct batch size
        B_dummy = 1
        # Only batch size matters when xdep=False
        feat_dummy = torch.empty(B_dummy, 0, device=device)
        pi, mu, cov = head(feat_dummy)   # pi:(1,K), mu:(1,K,D), cov depends

        K, D = mu.size(1), mu.size(2)
        # Build mixture distribution in latent space
        gmm = build_gmm_fn(pi, mu, cov, cov_type=args.cov_type)

        # Sample many z for smooth density
        z = gmm.sample((S_latent,))             # (S, 1, D)
        z = z.view(S_latent, D)                 # (S,D)

        # Latent PCA basis from component means (nice orientation)
        muK = mu[0]                             # (K,D)
        Xm_mu, P_mu, _ = _project_pca_2d(muK)
        Z2 = (z - Xm_mu) @ P_mu.T               # (S,2)

        # Distribution after g_B (projected perturbations)
        eps_flat = _latent_to_eps(
            z, use_decoder=(decoder is not None), decoder=decoder,
            C=C, H=H, W=W, gamma=args.gamma, norm_type=args.norm,
            g_ball_fn=g_ball_fn
        )                                       # (S, C*H*W)
        # PCA based on eps samples themselves (their geometry differs from z)
        _, _, Y_eps = _project_pca_2d(eps_flat)  # Y_eps: (S,2)

        # ---- plot & save ----
        lat_path = os.path.join(save_dir, "contour_latent_xindep.png")
        eps_path = os.path.join(save_dir, "contour_eps_xindep.png")

        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2))
        _contour_on_ax(ax, Z2, title="Latent (PCA) – x-independent", gridsize=gridsize)
        plt.tight_layout(); plt.savefig(lat_path); plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2))
        _contour_on_ax(ax, Y_eps, title="After g_B (PCA on ε) – x-independent", gridsize=gridsize)
        plt.tight_layout(); plt.savefig(eps_path); plt.close()

        print("[viz] saved:", lat_path, "and", eps_path)
        return lat_path, eps_path

    # -------- Case 2: x-dependent --------
    # collect the first n_inputs_when_xdep images
    xs, ys = [], []
    for xb, yb, _ in loader:
        xs.append(xb); ys.append(yb)
        if sum(x.shape[0] for x in xs) >= n_inputs_when_xdep:
            break
    Xall = torch.cat(xs, 0)[:n_inputs_when_xdep].to(device)   # (N,C,H,W)
    # Yall = torch.cat(ys, 0)[:n_inputs_when_xdep].to(device)  # labels not used here, keep if needed

    # Prepare subplots: latent contours and eps contours in separate figures
    n = Xall.size(0)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig_lat, axes_lat = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    fig_eps, axes_eps = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes_lat = np.array(axes_lat).reshape(-1) if isinstance(axes_lat, np.ndarray) else np.array([axes_lat])
    axes_eps = np.array(axes_eps).reshape(-1) if isinstance(axes_eps, np.ndarray) else np.array([axes_eps])

    for i in range(n):
        x_i = Xall[i:i+1]                      # (1,C,H,W)
        # forward encoder
        feat_i = encoder(x_i)                  # (1, feat_dim)

        pi, mu, cov = head(feat_i)             # (1,K,*)
        K, D = mu.size(1), mu.size(2)

        # build latent mixture and sample
        gmm_i = build_gmm_fn(pi, mu, cov, cov_type=args.cov_type)
        z = gmm_i.sample((S_latent,)).view(S_latent, D)  # (S,D)

        # latent PCA from component means at this x_i
        muK = mu[0]                            # (K,D)
        Xm_mu, P_mu, _ = _project_pca_2d(muK)
        Z2 = (z - Xm_mu) @ P_mu.T              # (S,2)

        # eps after g_B
        eps_flat = _latent_to_eps(
            z, use_decoder=(decoder is not None), decoder=decoder,
            C=C, H=H, W=W, gamma=args.gamma, norm_type=args.norm,
            g_ball_fn=g_ball_fn
        )                                      # (S, C*H*W)
        _, _, Y_eps = _project_pca_2d(eps_flat)   # (S,2)

        # draw to axes
        _contour_on_ax(axes_lat[i], Z2,   title=f"Latent (PCA) – sample {i}", gridsize=gridsize)
        _contour_on_ax(axes_eps[i], Y_eps, title=f"After g_B (PCA on ε) – sample {i}", gridsize=gridsize)

    # turn off unused axes (if any)
    for j in range(n, len(axes_lat)):
        axes_lat[j].axis("off")
    for j in range(n, len(axes_eps)):
        axes_eps[j].axis("off")

    fig_lat.tight_layout(); fig_eps.tight_layout()
    lat_path = os.path.join(save_dir, "contour_latent_xdep.png")
    eps_path = os.path.join(save_dir, "contour_eps_xdep.png")
    fig_lat.savefig(lat_path); fig_eps.savefig(eps_path)
    plt.close(fig_lat); plt.close(fig_eps)

    print("[viz] saved:", lat_path, "and", eps_path)
    return lat_path, eps_path

