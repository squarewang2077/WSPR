import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_tensor_bars(
    t: torch.Tensor,
    *,
    titles=None,          
    xlabel: str = None,
    ylabel: str = None,
    xticks_step: int = 1,  
    col_wrap: int = 3,    
    sharey: bool = True,  
    bar_width: float = 0.8,
    color: str = "0.2",   # 默认纯灰
    cmap: str = None,     # 新增：colormap 名称，如 "Blues", "viridis", "coolwarm"
    figsize=None,         
):
    """
    画 torch.tensor 的柱状图（[K] 或 [K, D]）。
    支持 cmap 控制配色。
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    x = t.detach().float().cpu().numpy()

    def _get_colors(values):
        """根据 cmap 生成颜色，否则用固定颜色"""
        if cmap is None:
            return color
        norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
        return cm.get_cmap(cmap)(norm(values))

    if x.ndim == 1:
        K = x.shape[0]
        if figsize is None:
            figsize = (max(3.0, min(0.35 * K + 1.5, 9.0)), 3.2)

        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        _style_axes(ax)
        ax.bar(
            np.arange(K),
            x,
            width=bar_width,
            color=_get_colors(x),
            edgecolor="0.0",
            linewidth=0.6
        )
        ax.set_xticks(np.arange(0, K, xticks_step))
        ax.set_xticklabels([str(i+1) for i in range(0, K, xticks_step)])
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if titles: ax.set_title(str(titles), fontsize=10, pad=6)
        return fig, ax

    elif x.ndim == 2:
        K, D = x.shape
        ncols = max(1, int(col_wrap))
        nrows = math.ceil(K / ncols)
        if figsize is None:
            per_w, per_h = 3.0, 2.6
            figsize = (per_w * ncols, per_h * nrows)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharey=sharey, squeeze=False, constrained_layout=True
        )

        for k in range(K):
            r, c = divmod(k, ncols)
            ax = axes[r][c]
            _style_axes(ax)
            ax.bar(
                np.arange(D),
                x[k],
                width=bar_width,
                color=_get_colors(x[k]),
                edgecolor="0.0",
                linewidth=0.6
            )
            ax.set_xticks(np.arange(0, D, xticks_step))
            ax.set_xticklabels([str(i+1) for i in range(0, D, xticks_step)])
            if ylabel and (c == 0):
                ax.set_ylabel(ylabel)
            if xlabel and (r == nrows - 1):
                ax.set_xlabel(xlabel)
            if titles is not None:
                if isinstance(titles, (list, tuple)) and len(titles) == K:
                    ax.set_title(str(titles[k]), fontsize=10, pad=6)
                else:
                    ax.set_title(f"Group {k+1}", fontsize=10, pad=6)

        for k in range(K, nrows * ncols):
            r, c = divmod(k, ncols)
            fig.delaxes(axes[r][c])

        return fig, axes
    else:
        raise ValueError("Tensor must be 1D [K] or 2D [K, D].")


def _style_axes(ax):
    """极简黑白风格，适合论文。"""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("0.0")

    ax.tick_params(axis="both", which="both", length=3, width=0.8, color="0.0", labelsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.6", alpha=0.8)
    ax.set_axisbelow(True)


def pick_projection(Sigma_stack: torch.Tensor, pi: torch.Tensor, max_dim: int = 32):
    """
    Sigma_stack: (K,D,D), pi:(K,)
    return:
      P_d  : (D, d)   d = min(D, max_dim)  —— for latent d-dim heatmap & cov/corr heatmap
      P_2d : (D, 2)   —— for latent 2D contour
    """
    K, D, _ = Sigma_stack.shape
    d = min(D, max_dim)
    if D <= d:
        P_d  = torch.eye(D, device=Sigma_stack.device)
        # 若 D==1，只能扩一维做图
        if D >= 2:
            P_2d = torch.eye(D, device=Sigma_stack.device)[:, :2]
        else:
            P_2d = torch.eye(1, device=Sigma_stack.device)
        return P_d, P_2d

    w = (pi / (pi.sum() + 1e-12)).view(K, 1, 1) # for stability
    Sigma_avg = (w * Sigma_stack).sum(dim=0)              # (D,D)
    evals, evecs = torch.linalg.eigh(Sigma_avg)           # ascending order
    P_d  = evecs[:, -d:]                                  # top d eigenvectors
    P_2d = evecs[:, -2:] if D >= 2 else evecs[:, -1:].clone()
    return P_d, P_2d


def cov_to_full(cov, cov_type: str) -> torch.Tensor:
    if cov_type == "diag":
        sigma = cov  # (K,D)
        K, D = sigma.shape
        eye = torch.eye(D, device=sigma.device).unsqueeze(0)
        return (sigma ** 2).unsqueeze(-1) * eye
    elif cov_type == "full":
        L = cov  # (K,D,D)
        return L @ L.transpose(-1, -2)
    elif cov_type == "lowrank":
        U, sigma = cov  # U:(K,D,r), sigma:(K,D)
        K, D, _ = U.shape
        eye = torch.eye(D, device=U.device).unsqueeze(0)
        return U @ U.transpose(-1, -2) + (sigma ** 2).unsqueeze(-1) * eye + 1e-6 * eye
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")


### for heatmap

def plot_tensor_heatmaps(
    t: torch.Tensor,                 # [K, D, D]
    *,
    titles=None,                     # 长度为 K 的标题（可含 LaTeX）
    nrows: int | None = None,        # 行数（指定其一或二者皆可）
    ncols: int | None = None,        # 列数
    share_color: bool = True,        # 统一 vmin/vmax
    vmin: float = None,
    vmax: float = None,
    cmap: str = "gray",              # 基础 colormap
    reverse_cmap: bool = False,      # 是否反转配色（colorbar 颜色对调）
    xtick_step: int = None,          # 刻度间隔
    show_colorbar: bool = True,
    figsize=None,
    square: bool = True,
):
    """
    绘制 [K, D, D] 的张量为 K 张热图。
    返回: (fig, axes, images)
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    x = t.detach().float().cpu().numpy()
    if x.ndim != 3 or x.shape[1] != x.shape[2]:
        raise ValueError("Tensor must have shape [K, D, D].")

    K, D, _ = x.shape

    # colormap（可反转）
    base_cmap = plt.cm.get_cmap(cmap)
    if reverse_cmap:
        base_cmap = base_cmap.reversed()

    # 统一色标范围
    if share_color:
        vmin = np.min(x) if vmin is None else vmin
        vmax = np.max(x) if vmax is None else vmax

    # 排版：按 nrows/ncols 指定；若缺省则尽量方阵
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(K)))
        nrows = int(np.ceil(K / ncols))
    elif nrows is None:
        nrows = int(np.ceil(K / ncols))
    elif ncols is None:
        ncols = int(np.ceil(K / nrows))
    if nrows * ncols < K:
        raise ValueError("nrows * ncols must be >= K")

    if figsize is None:
        per_w, per_h = 2.8, 2.8
        figsize = (per_w * ncols, per_h * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )

    images = []
    # 自动刻度密度
    if xtick_step is None:
        xtick_step = 1 if D <= 20 else max(1, D // 10)

    for k in range(K):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        _style_axes_heat(ax)
        im = ax.imshow(
            x[k], cmap=base_cmap, vmin=vmin, vmax=vmax,
            aspect="equal" if square else "auto", interpolation="nearest"
        )
        images.append(im)

        ticks = np.arange(0, D, xtick_step)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([str(i+1) for i in ticks], fontsize=8)
        ax.set_yticklabels([str(i+1) for i in ticks], fontsize=8)

        if r == nrows - 1: ax.set_xlabel("", fontsize=9)
        if c == 0:         ax.set_ylabel("", fontsize=9)

        if titles is not None:
            if isinstance(titles, (list, tuple)) and len(titles) == K:
                ax.set_title(str(titles[k]), fontsize=10, pad=4)
            else:
                ax.set_title(f"Cov. {k+1}", fontsize=10, pad=4)

    # 删除空白子图
    for k in range(K, nrows * ncols):
        r, c = divmod(k, ncols)
        fig.delaxes(axes[r][c])

    if show_colorbar:
        valid_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)
                      if (r * ncols + c) < K]
        fig.colorbar(images[0] if share_color else images[-1],
                     ax=valid_axes, fraction=0.025, pad=0.02)

    return fig, axes, images


def _style_axes_heat(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("0.0")
    ax.tick_params(axis="both", which="both", length=3, width=0.8, color="0.0", labelsize=8)



# def _to_numpy(x: torch.Tensor):
#     return x.detach().cpu().numpy()


# def _to_correlation(Sigma: torch.Tensor) -> torch.Tensor:
#     eps = 1e-8
#     var = Sigma.diagonal(dim1=-2, dim2=-1).clamp_min(eps)  # (K,D)
#     std = var.sqrt()
#     denom = std.unsqueeze(-1) * std.unsqueeze(-2)          # (K,D,D)
#     Corr = Sigma / denom
#     return Corr.clamp_(-1.0, 1.0)


# @torch.no_grad()
# def _kde2d(points_2d: torch.Tensor,
#            xmin: float, xmax: float, ymin: float, ymax: float,
#            gridsize: int = 200, bandwidth: float = None):
#     device = points_2d.device
#     N = points_2d.size(0)
#     if N == 0:
#         raise ValueError("kde2d: empty input.")
#     if bandwidth is None:
#         std = points_2d.std(dim=0).clamp(min=1e-6)
#         h = 1.06 * std.mean() * (float(N) ** (-1.0/5.0))
#     else:
#         h = torch.tensor(float(bandwidth), device=device)

#     xs = torch.linspace(xmin, xmax, gridsize, device=device)
#     ys = torch.linspace(ymin, ymax, gridsize, device=device)
#     X, Y = torch.meshgrid(xs, ys, indexing="xy")
#     Px = points_2d[:, 0].view(-1, 1, 1)
#     Py = points_2d[:, 1].view(-1, 1, 1)
#     Z = torch.exp(-((X.unsqueeze(0)-Px)**2 + (Y.unsqueeze(0)-Py)**2)/(2*h*h)).mean(0)
#     Z = Z / Z.max().clamp_min(1e-12)
#     return X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()

# def _contour_plot(ax, pts2d: torch.Tensor, title: str, gridsize=200, pad=0.1):
#     if pts2d.numel() == 0:
#         ax.set_title(title + " (no data)"); ax.axis("off"); return
#     x = pts2d[:,0]; y=pts2d[:,1]
#     xmin, xmax = float(x.min()), float(x.max())
#     ymin, ymax = float(y.min()), float(y.max())
#     rx, ry = xmax-xmin, ymax-ymin
#     if rx <= 0: rx = 1.0
#     if ry <= 0: ry = 1.0
#     xmin -= pad*rx; xmax += pad*rx
#     ymin -= pad*ry; ymax += pad*ry
#     X, Y, Z = _kde2d(pts2d, xmin, xmax, ymin, ymax, gridsize=gridsize)
#     ax.contourf(X, Y, Z, levels=12, alpha=0.85)
#     ax.contour(X, Y, Z, levels=12, linewidths=0.8)
#     ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

# def _latent_to_eps(z: torch.Tensor, use_decoder: bool, decoder,
#                    C: int, H: int, W: int,
#                    gamma: float, norm_type: str,
#                    g_ball_fn):
#     N, D = z.shape
#     if use_decoder:
#         u = decoder(z)
#         if u.dim() == 2:
#             u = u.view(N, C, H, W)
#     else:
#         u = z.view(N, C, H, W)
#     eps = g_ball_fn(u, gamma=gamma, norm_type=norm_type)
#     return eps.view(N, -1)


# @torch.no_grad()
# def visualize_gmm_components_notebook(
#     head,
#     *,
#     encoder=None,                 # xdep=True need encoder
#     x_for_xdep: torch.Tensor=None,
#     # take from your model if use_decoder=True
#     out_shape=None,               # (C,H,W)，needed when use_decoder=True
#     g_ball_fn=None,
#     use_decoder: bool=False,
#     gamma: float=8/255,
#     norm_type: str="linf",
#     # plot settings
#     as_correlation: bool=True,
#     max_cov_dim: int=32,
#     S_latent: int=4000,           # for every contour sampling number
#     gridsize: int=200,
#     figure_dpi: int=160,
#     save_dir: str=None            # if not None，save in the directory
# ):
#     """
#     Generate an individual image for each Gaussian component k (and display in the notebook):
#       (1) π_k small bar plot
#       (2) μ_k 1xd heatmap in the shared d-dimensional subspace
#       (3) Σ_k or correlation matrix dxd heatmap in that subspace
#       (4) Latent 2D contour (using the first two dimensions of the global projection basis)
#       (5) 2D contour after g_B (perform PCA->2D on ε samples)
#     """
#     device = next(head.parameters()).device if hasattr(head, "parameters") else torch.device("cpu")

#     # take (pi, mu, cov) from head
#     if getattr(head, "xdep", False):
#         if encoder is None or x_for_xdep is None:
#             raise RuntimeError("x-dependent GMM 需要 encoder 和 x_for_xdep。")
#         x = x_for_xdep
#         if x.dim() == 3: x = x.unsqueeze(0)
#         feat = encoder(x.to(device))
#     else:
#         feat = torch.empty(1, 0, device=device)

#     pi_b, mu_b, cov_b = head(feat) # pi:(1,K), mu:(1,K,D)
#     pi = pi_b[0]; mu = mu_b[0]  # (K,), (K,D)

#     # ---- setup of full Sigma ----
#     if head.cov_type == "diag":
#         Sigma = _cov_to_full(cov_b[0], "diag") # (K,D,D)
#     elif head.cov_type == "full":
#         Sigma = _cov_to_full(cov_b[0], "full")
#     else:
#         U, sigma = cov_b
#         Sigma = _cov_to_full((U[0], sigma[0]), "lowrank")

#     K, D = mu.shape

#     # ---- shared projection base：P_d (D->d), P_2d (D->2) ----
#     P_d, P_2d = pick_projection(Sigma, pi, max_dim=max_cov_dim)
#     d = P_d.shape[1]

#     # ---- setup of color range ----
#     mu_proj_all = mu @ P_d # (K,d)
#     mu_abs = float(mu_proj_all.abs().max().item())
#     mu_vmax = max(1e-6, mu_abs); mu_vmin = -mu_vmax

#     # cov/corr matrix in the projected d-dim space
#     Sigma_proj = torch.stack([P_d.t() @ Sigma[k] @ P_d for k in range(K)], dim=0)  # (K,d,d)
#     if as_correlation:
#         Mat_stack = _to_correlation(Sigma_proj)
#         cov_cmap, cov_vmin, cov_vmax, cov_label = "coolwarm", -1.0, 1.0, "Correlation"
#     else:
#         Mat_stack = Sigma_proj
#         ma = float(Mat_stack.abs().max().item())
#         cov_vmax = max(1e-6, ma); cov_vmin = -cov_vmax
#         cov_cmap, cov_label = "coolwarm", "Covariance"

#     # ---- for saving ----
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)

#     C = H = W = None
#     if use_decoder:
#         if out_shape is None:
#             raise ValueError("use_decoder=True 时必须提供 out_shape=(C,H,W).")
#         C, H, W = out_shape
#         if g_ball_fn is None:
#             raise ValueError("use_decoder=True 时必须提供 g_ball_fn.")

#     for k in range(K):
#         fig = plt.figure(figsize=(12, 6.5), dpi=figure_dpi)
#         gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1,1.3])

#         # (1) π_k
#         ax_pi = fig.add_subplot(gs[0, 0])
#         ax_pi.bar([0], [float(pi[k])], width=0.6)
#         ax_pi.set_title(f"π[{k}] = {float(pi[k]):.4f}")
#         ax_pi.set_xticks([]); ax_pi.set_ylabel("Weight")
#         ax_pi.set_ylim(0, float(pi.max())*1.05 + 1e-6)

#         # (2) μ_k heatmap（1×d）
#         ax_mu = fig.add_subplot(gs[0, 1:4])
#         mu_k_proj = (mu[k] @ P_d).unsqueeze(0)           # (1,d)
#         im_mu = ax_mu.imshow(_to_numpy(mu_k_proj), aspect="auto",
#                              cmap="coolwarm", vmin=mu_vmin, vmax=mu_vmax)
#         ax_mu.set_title("Projected mean (1×d)")
#         ax_mu.set_yticks([]); ax_mu.set_xlabel(f"Projected dim (d={d})")
#         cbar_mu = fig.colorbar(im_mu, ax=ax_mu, fraction=0.025, pad=0.02)
#         cbar_mu.ax.set_ylabel("Mean value", rotation=90)

#         # (3) Σ_k / Corr_k heatmap（d×d）
#         ax_cov = fig.add_subplot(gs[1, 0])
#         im_cov = ax_cov.imshow(_to_numpy(Mat_stack[k]), cmap=cov_cmap, vmin=cov_vmin, vmax=cov_vmax)
#         ax_cov.set_title(f"{cov_label} (d×d)")
#         ax_cov.set_xticks([]); ax_cov.set_yticks([])
#         cbar_cov = fig.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.02)

#         # (4) latent 2D use P_2d
#         ax_lat = fig.add_subplot(gs[1, 1])
#         # 用该分量的 full cov 构造一个 2D 采样：先在 D 维采样，再投到 2D
#         jitter = 1e-6
#         cov_full = Sigma[k] + jitter * torch.eye(D, device=device)
#         mvn_k = torch.distributions.MultivariateNormal(loc=mu[k], covariance_matrix=cov_full)
#         z = mvn_k.sample((S_latent,))                    # (S,D)
#         z2 = z @ P_2d                                    # (S,2)
#         _contour_plot(ax_lat, z2, title="Latent (2D KDE)", gridsize=gridsize)

#         # (5) after g_B contour（对 ε 做 PCA->2D 再 KDE）
#         ax_eps = fig.add_subplot(gs[1, 2])
#         if use_decoder:
#             eps_flat = _latent_to_eps(
#                 z, use_decoder=True, decoder=encoder if hasattr(encoder, "__call__") and encoder is not None and False else head,  # 占位不会用到
#                 C=C, H=H, W=W, gamma=gamma, norm_type=norm_type, g_ball_fn=g_ball_fn
#             )
#             # 上面那行只是满足签名占位，不会调用 head；真正应当使用你的 decoder/g_ball_fn：
#         else:
#             # 无 decoder：直接把 z 当作 u，再投影到球得到 eps
#             if g_ball_fn is None:
#                 raise ValueError("未使用 decoder 时也请提供 g_ball_fn 以投影到 L_p 球。")
#             eps_flat = _latent_to_eps(z, use_decoder=False, decoder=None,
#                                       C=1, H=1, W=D, gamma=gamma, norm_type=norm_type, g_ball_fn=g_ball_fn)

#         # 对 eps_flat 做 2D PCA
#         X = eps_flat - eps_flat.mean(0, keepdim=True)
#         U, S, Vt = torch.linalg.svd(X, full_matrices=False)
#         P2_eps = Vt[:2, :] if X.shape[1] >= 2 else torch.eye(1, device=X.device)
#         Y2 = X @ P2_eps.T if X.shape[1] >= 2 else torch.cat([X, torch.zeros(X.size(0),1,device=X.device)], dim=1)
#         _contour_plot(ax_eps, Y2[:, :2], title="After g_B (2D KDE)", gridsize=gridsize)

#         # 右下角空出来放额外说明
#         ax_txt = fig.add_subplot(gs[1, 3])
#         ax_txt.axis("off")
#         ax_txt.text(0.0, 0.9, f"Component k = {k}", fontsize=11)
#         ax_txt.text(0.0, 0.7, f"D = {D}, d = {d}", fontsize=10)
#         ax_txt.text(0.0, 0.55, f"S_latent = {S_latent}", fontsize=10)
#         ax_txt.text(0.0, 0.4, f"norm = {norm_type}, gamma = {gamma}", fontsize=10)

#         fig.suptitle(f"GMM component #{k}", y=0.98)
#         plt.show()

#         if save_dir is not None:
#             path = os.path.join(save_dir, f"gmm_component_{k:02d}.png")
#             fig.savefig(path, bbox_inches="tight")
#             print(f"[viz] saved -> {path}")
