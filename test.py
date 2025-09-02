# file: wspr_gmm_optionA.py
import os, argparse, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from attacks import WithIndex

# -------------------- Head: outputs GMM params --------------------
class GMMHead(nn.Module):
    def __init__(self, feat_dim=512, K=3, D=3072, xdep=True):
        super().__init__()
        self.K, self.D, self.xdep = K, D, xdep
        if xdep:
            self.trunk = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU())
            self.pi_head   = nn.Linear(256, K)          # (B,K)
            self.mu_head   = nn.Linear(256, K*D)        # (B,K*D)
            self.logsig_head = nn.Linear(256, K*D)      # (B,K*D)
        else:
            # 全局常量（与 x 无关）
            self.pi_logits   = nn.Parameter(torch.zeros(K))
            self.mu          = nn.Parameter(torch.zeros(K, D))
            self.log_sigma   = nn.Parameter(torch.zeros(K, D))

    def forward(self, feat=None):
        if self.xdep:
            h = self.trunk(feat)                                 # (B,256)
            pi = F.softmax(self.pi_head(h), dim=-1)              # (B,K)
            mu = self.mu_head(h).view(-1, self.K, self.D)        # (B,K,D)
            sigma = self.logsig_head(h).view(-1, self.K, self.D).exp().clamp_min(1e-6)
        else:
            B = feat.size(0) if feat is not None else 1
            pi = self.pi_logits.softmax(-1).expand(B, -1)        # (B,K)
            mu = self.mu.unsqueeze(0).expand(B, -1, -1)          # (B,K,D)
            sigma = self.log_sigma.exp().clamp_min(1e-6).unsqueeze(0).expand(B, -1, -1)
        return pi, mu, sigma

# -------------------- Build torch.distributions GMM (for eval/WSPR) --------------------
def build_gmm(pi, mu, sigma):
    # pi:(B,K), mu/sigma:(B,K,D)
    mix = Categorical(pi)                               # mixture
    comp = Independent(Normal(mu, sigma), 1)            # (B,K,D) vector-component
    return MixtureSameFamily(mix, comp)                 # 注意：没有 rsample！

# -------------------- Option A loss: pathwise per component --------------------
def optionA_loss(net, pi, mu, sigma, x, y, gamma=8/255, S=1):
    """
    L = sum_k pi_k(x) * E_z CE(f(x + g_B(mu_k + sigma_k * z)), y),  z~N(0,I)
    这里对每个分量使用 Normal.rsample，避免对混合 rsample（MixtureSameFamily 没实现）。
    """
    device = x.device
    B, C, H, W = x.shape
    K, D = mu.size(1), mu.size(2)
    total = 0.0
    for k in range(K):
        # (S,B,D) 路径采样
        z = torch.randn(S, B, D, device=device)
        eps_latent = mu[:, k, :].unsqueeze(0) + sigma[:, k, :].unsqueeze(0) * z
        # 直接视作像素噪声（D=C*H*W）
        eps = eps_latent.view(S*B, C, H, W)
        eps = torch.tanh(eps) * gamma                      # g_B
        x_rep = x.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S*B, C, H, W)
        y_rep = y.repeat(S)
        logits = net(x_rep + eps)
        L = F.cross_entropy(logits, y_rep, reduction="none").view(S, B).mean(0) # E_z
        # 权重加和（pi 可能是 (B,K)）
        total = total + pi[:, k] * L
    return total.mean()

# -------------------- Train phi (GMM head) --------------------
def fit_phi(net, loader, args, device):
    # 冻结分类器；用其 backbone 提特征（最后一层 avgpool + flatten）
    backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten()).to(device)
    for p in net.parameters(): p.requires_grad_(False)
    for p in backbone.parameters(): p.requires_grad_(False)

    C,H,W = 3,32,32
    D = C*H*W
    head = GMMHead(feat_dim=512, K=args.num_modes, D=D, xdep=args.xdep).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        for it,(x,y,_) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            feat = backbone(x)                               # (B,512)
            pi, mu, sigma = head(feat)                       # (B,K),(B,K,D),(B,K,D)
            loss = optionA_loss(net, pi, mu, sigma, x, y, gamma=args.gamma, S=args.mc)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f"[ep {ep} it {it}] loss={loss.item():.4f} | pi[0]={pi[0].detach().cpu().numpy()}")
    return head, backbone

# -------------------- Estimate WSPR (Monte Carlo under learned GMM) --------------------
@torch.no_grad()
def estimate_wspr(net, gmm, x, y, gamma=8/255, T=50):
    """
    估计  E[ 1{ f(x+g_B(eps)) = y } ]  ，eps ~ GMM_phi(x)
    这里用 mixture.sample（评估时可用，无需梯度），然后过 g_B 与分类器。
    """
    B, C, H, W = x.shape
    # (T,B,D)
    eps_latent = gmm.sample((T,))
    eps = torch.tanh(eps_latent.view(T, B, C, H, W)) * gamma
    x_rep = x.unsqueeze(0) + eps                           # (T,B,C,H,W)
    logits = net(x_rep.view(T*B, C, H, W)).view(T, B, -1)
    pred = logits.argmax(dim=-1)                           # (T,B)
    correct = (pred == y.unsqueeze(0)).float().mean(dim=0) # per-sample prob
    return correct.mean().item()                           # batch average

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_modes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=8/255)
    ap.add_argument("--mc", type=int, default=1)
    ap.add_argument("--xdep", action="store_true", help="让 π,μ,σ 依赖 x")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据：用 CIFAR-10 test，与你原脚本一致
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    testset = WithIndex(torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=tf))
    loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # 分类器：这里用随机初始化（你也可以自己 load 预训练权重）
    net = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    net.eval(); [p.requires_grad_(False) for p in net.parameters()]

    # 训练 φ（得到 x->GMM 的映射）
    head, backbone = fit_phi(net, loader, args, device)

    # —— 如何拿到 Distribution 对象并估计 WSPR —— #
    x, y, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    pi, mu, sigma = head(backbone(x))
    gmm = build_gmm(pi, mu, sigma)                   # 这是正式的 torch.distributions GMM
    wspr_est = estimate_wspr(net, gmm, x, y, gamma=args.gamma, T=32)
    print(f"[eval] estimated WSPR (prob. of correct under learned GMM): {wspr_est:.4f}")

if __name__ == "__main__":
    main()
