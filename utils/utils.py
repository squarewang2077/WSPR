import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from attacks import WithIndex
from tqdm import tqdm

### -------------------- Dataset -------------------- ###
def get_dataset(name, root="./dataset", train=False):
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
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf))
        
        num_classes = 10
        input_shape = (3,32,32)

    elif name == "cifar100":
        mean, std = (0.5071,0.4865,0.4409), (0.2673,0.2564,0.2762)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=tf))

        num_classes = 100
        input_shape = (3,32,32)

    elif name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.MNIST(root=root, train=train, download=True, transform=tf))

        num_classes = 10
        input_shape = (1,28,28)

    elif name == "tinyimagenet":
        mean, std = (0.4802,0.4481,0.3975), (0.2302,0.2265,0.2262)
        tf = T.Compose([T.Resize(64),T.CenterCrop(64),T.ToTensor(),T.Normalize(mean,std)])
        ds = WithIndex(torchvision.datasets.ImageFolder(os.path.join(root,"tiny-imagenet-200","val"), transform=tf))

        num_classes = 200
        input_shape = (3,64,64)

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