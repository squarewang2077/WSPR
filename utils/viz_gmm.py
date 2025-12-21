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
