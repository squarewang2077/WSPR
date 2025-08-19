import torch
import random
from einops import rearrange
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class WithIndex(torch.utils.data.Dataset):
    """
    Wrap an existing dataset so __getitem__ returns (..., idx).
    Works whether the base dataset returns (img, label) or a dict.
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        if isinstance(item, dict):
            item = dict(item)
            item["idx"] = idx
            return item
        elif isinstance(item, (list, tuple)):
            return (*item, idx)
        else:
            # If dataset returns only img, we attach idx and a dummy label? Better to ensure (img,label)
            # Here we just return (item, None, idx) for completeness.
            return (item, None, idx)

@torch.no_grad()
def n_attack_perbatch(model = None, imgs = None, labels = None, N_noise = 300, sigma = 0.1, \
             att_type = 'infty', epsi = 0.031, \
             N_step = 500, step_size = 0.02, reward_scaling = 0.5,\
             img_mean = [0.4914, 0.4822, 0.4465], img_std = [0.2023, 0.1994, 0.2010]):

    """    
    N_attack implement on a given model.
    Args:
        model (torch.nn.Module): The target model to attack
        imgs (torch.Tensor): Input images to attack, shape (B, C, H, W)
        ground_truth (int): The ground truth label for the input image, shape (B, Label)
        N_noise (int): Number of gaussian noise samples to use for the attack
        N_step (int): Number of steps to run the attack
        step_size (float): Step size for the attack
        sigma (float): Standard deviation of the gaussian noise
        att_type (str): Type of norm to use ('infty' or 'l2')
        epsi (float): Perturbation budget
        reward_scaling (float): Scaling factor for the reward
        eps (float): Small value to avoid division by zero

    """
    # check whether all the imgs are correctly classified
    def check_classification():
        with torch.no_grad():
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct = (preds == labels).float().sum()
            print(f"Correctly classified {correct}/{imgs.shape[0]} images.")

    check_classification()

    # some imgs have be normalized before feed into the model, we need to inverse the normalization
    img_mean = torch.tensor(img_mean, dtype=imgs.dtype, device=imgs.device).view(1,1,3,1,1) # (B, N, C, H, W)
    img_std  = torch.tensor(img_std, dtype=imgs.dtype, device=imgs.device).view(1,1,3,1,1)

    # extend the input image to a batch of size 1
    imgs = imgs.unsqueeze(1) # (B,1,C,H,W)


    def arctanh_torch(x: torch.Tensor) -> torch.Tensor:
    # stable arctanh on (-1, 1)
        return 0.5 * torch.log((1 + x + 1e-12) / (1 - x + 1e-12))

    def img_transform(imgs: torch.Tensor) -> torch.Tensor:
        """Transform the input image to tanh-space."""
        # imgs: (3, 32, 32), float tensor, conduct inver-normalization
        imgs = imgs * img_std + img_mean # range of [0,1]
        imgs = arctanh_torch((imgs - 0.5) / 0.5) # range of (-inf, +inf)
        return imgs

    def img_inverse_transform(imgs: torch.Tensor) -> torch.Tensor:
        """Inverse transform from tanh-space to [0,1]"""
        imgs = torch.tanh(imgs) * 0.5 + 0.5 # range of [0,1]
        imgs = (imgs - img_mean) / img_std # original range of normalized image
        return imgs


    # transform the input image to tanh-space of -inft to +infty
    trans_img = img_transform(imgs) # (B,1,C,H,W)


    def mu_update_nes(imgs, N_step, step_size, sigma, N_noise):
        """
        Update the Gaussian noise mean using the Natural Evolution Strategy (NES) method.
        """

        # initialized perturbation with a small standard deviation and the same size of input image batch
        mu = torch.randn_like(imgs) * 1e-3 # (B,1,C,H,W)

        for _ in range(N_step):
            # draw samples of gassian noise for each batch of images [N_noise, C, H, W]
            samples = torch.randn((imgs.shape[0], N_noise, *imgs.shape[2:]), device=imgs.device, dtype=imgs.dtype) # [B, N_noise, C, H, W]
            noise_update = mu + sigma * samples # [B,1,C,H,W] + sigma * [B, N_noise, C, H, W]

            # candidate images (stay in [0,1])
            new_imgs = img_inverse_transform(trans_img + noise_update)  # (B,N_noise,C,H,W)

            delta = new_imgs - imgs
            if att_type == 'infty':
                # enforce L_inf
                delta = torch.clamp(delta, -epsi, epsi)
                clip_imgs = imgs + delta
            elif att_type == 'l2':
                # enforce L2 norm
                norm = torch.norm(delta.view(N_noise, -1), p=2, dim=1, keepdim=True)
                scale = torch.clamp(norm / epsi, min=1e-8)
                clip_imgs = imgs + delta / scale.view(N_noise, 1, 1)

            # evaluate batch
            logits = model(rearrange(clip_imgs, 'b n c h w -> (b n) c h w'))
            logits = rearrange(logits, '(b n) k -> b n k', b=clip_imgs.shape[0], n=clip_imgs.shape[1])

            # CW-style margin loss
            # logits: [B, N, K]   (B = batch size, N = #noises, K = #classes)
            # labels: [B]         (ground-truth class per image)
            B, N, K = logits.shape
            labels = labels.view(B)  # ensure shape

            # expand labels across the N noise samples: [B, N]
            labels_exp = labels[:, None].expand(B, N)

            # true-class logit z_y: [B, N]
            real = logits.gather(2, labels_exp.unsqueeze(-1)).squeeze(-1) # .gather(2, [8,300,1])

            # max non-true logit max_{i≠y} z_i
            mask = F.one_hot(labels, num_classes=K).bool()          # [B, K]
            mask = mask[:, None, :].expand(B, N, K)                 # [B, N, K]
            other = logits.masked_fill(mask, float('-inf')).amax(dim=2)   # [B, N] other is the second largest label

            # "Higher = more correct" CW-style margin on logits:
            margin = (real - other).clamp_min(0.0)                  # [B, N]

            # If you're ATTACKING (untargeted), use the opposite as reward:
            reward = reward_scaling * margin                                  # [B, N]
            A = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-7)

            # ES grad per image: sum over noise dim
            # einsum version:
            grad_bchw = torch.einsum('bnchw,bn->bchw', samples, A)   # (B,C,H,W)

            # Update noise (keep noise as (B,1,C,H,W))
            mu = mu + (step_size / (N_noise * sigma)) * grad_bchw.unsqueeze(1)

    mu = mu_update_nes(imgs, N_step, step_size, sigma, N_noise)

    return mu


def nattack(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    fraction: float = 1.0,
    device: Optional[torch.device | str] = None,  # KEEP IT NONE
    save_perturbations: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Attack a random fraction of samples from `dataloader`, but only those
    that are correctly classified. Record ONLY dataset indices where the attack succeeds.
    Expects each batch to include indices (either (imgs, labels, idxs) or dict with keys).
    Uses the batched attacker `n_attack_perbatch` for efficiency.
    """

    # clamp fraction
    fraction = max(0.0, min(1.0, float(fraction)))
    # NO NEED TO CHANGE device
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model.eval()

    succeeded_indices: List[int] = []
    saved_perturbations: Optional[List[torch.Tensor]] = [] if save_perturbations else None

    total_seen = 0 # total data points from all batches
    sampled = 0 # total data points sampled from all batches
    correct_on_clean = 0
    attacked = 0
    successes = 0
    skipped_misclassified = 0

    for batch in dataloader:
        # --- Extract imgs, labels, idxs robustly ---
        if isinstance(batch, dict):
            imgs = batch.get("image") or batch.get("images") or batch.get("x")
            labels = batch.get("label") or batch.get("labels") or batch.get("y")
            idxs = batch.get("idx") or batch.get("indices")
        elif isinstance(batch, (list, tuple)):
            if len(batch) < 3:
                raise ValueError(
                    "This nattack expects the dataloader to yield indices. "
                    "Wrap your dataset with WithIndex(...) so batches are (imgs, labels, idxs)."
                )
            imgs, labels, idxs = batch[0], batch[1], batch[2]
        else:
            raise ValueError(
                "Unsupported batch type. Provide (imgs, labels, idxs) or a dict with 'image','label','idx'."
            )

        if imgs is None or labels is None or idxs is None:
            raise ValueError(
                "Missing imgs/labels/idxs in batch. Wrap your dataset with WithIndex(...) "
                "or adjust keys to ('image','label','idx')."
            )

        # make idxs a simple CPU LongTensor of shape [B]
        if torch.is_tensor(idxs):
            idx_tensor = idxs.detach().cpu().long()
        else:
            idx_tensor = torch.as_tensor(list(idxs), dtype=torch.long)

        B = idx_tensor.numel()
        total_seen += B

        # --- Fractional sampling on CPU to avoid unnecessary transfers ---
        if fraction < 1.0:
            keep_mask = (torch.rand(B) < fraction)
        else:
            keep_mask = torch.ones(B, dtype=torch.bool)

        if keep_mask.sum().item() == 0: # for extreme smaller fractions, it can be 0, hence just continue 
            continue

        # Subset the batch by the fraction mask
        imgs_keep = imgs[keep_mask]
        labels_keep = labels[keep_mask]
        idx_keep = idx_tensor[keep_mask]  # CPU

        sampled += int(keep_mask.sum().item())

        # Move selected tensors to device
        imgs_keep = imgs_keep.to(device, non_blocking=True)
        labels_keep = labels_keep.to(device, non_blocking=True)

        # ---- Run the batched attacker (which itself filters to correctly classified) ----
        # Expected to be defined elsewhere: n_attack_perbatch(model, imgs_keep, labels_keep, **kwargs)
        out = n_attack_perbatch(model, imgs_keep, labels_keep, **kwargs)

        # Out fields: "delta" [B_k,C,H,W], "success" [B_k], "attacked_mask" [B_k] (bool)
        succ_vec = out["success"]                    # on device
        att_mask = out["attacked_mask"]              # on device/bool
        deltas   = out["delta"]                      # [B_k,C,H,W], on device

        num_attacked = int(att_mask.sum().item())
        num_success  = int(succ_vec.sum().item())

        correct_on_clean += num_attacked
        attacked        += num_attacked
        successes       += num_success
        skipped_misclassified += int(idx_keep.numel()) - num_attacked

        # Successful indices (subset of idx_keep)
        succ_cpu_mask = succ_vec.detach().cpu().bool()
        succ_indices  = idx_keep[succ_cpu_mask].tolist()
        succeeded_indices.extend(succ_indices)

        if save_perturbations:
            # save only perturbations for successful items
            for d in deltas[succ_vec.bool()].detach().cpu():
                saved_perturbations.append(d)

    success_rate = (successes / attacked) if attacked > 0 else 0.0
    return {
        "succeeded_indices": succeeded_indices,
        "success_rate": float(success_rate),
        "counts": {
            "total_seen": int(total_seen),
            "sampled": int(sampled),
            "correct_on_clean": int(correct_on_clean),
            "attacked": int(attacked),
            "successes": int(successes),
            "skipped_misclassified": int(skipped_misclassified),
        },
        "perturbations": saved_perturbations,  # None unless save_perturbations=True
    }
