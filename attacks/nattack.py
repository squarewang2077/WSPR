import torch
import random
from einops import rearrange
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils



@torch.no_grad()
def n_attack_perbatch(model = None, imgs = None, labels = None, N = 300, sigma = 0.1, \
             att_type = 'infty', epsi = 0.031, \
             N_step = 500, step_size = 0.02, reward_scaling = 0.5,\
             img_mean = [0.4914, 0.4822, 0.4465], img_std = [0.2023, 0.1994, 0.2010]):

    """    
    N_attack implement on a given model.
    Args:
        model (torch.nn.Module): The target model to attack
        imgs (torch.Tensor): Input images to attack, shape (B, C, H, W)
        ground_truth (int): The ground truth label for the input image, shape (B, Label)
        N (int): Number of gaussian noise samples to use for the attack
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
            print(f"Correctly classified {int(correct)}/{imgs.shape[0]} images.")

    check_classification()

    # some imgs have be normalized before feed into the model, we need to inverse the normalization
    B, C, H, W = imgs.shape
    img_mean = torch.tensor(img_mean, dtype=imgs.dtype, device=imgs.device).view(1,1,C,1,1) # (B, N, C, H, W)
    img_std  = torch.tensor(img_std, dtype=imgs.dtype, device=imgs.device).view(1,1,C,1,1)

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

    ### Update the mu by NES ###
    # transform the input image to tanh-space of -inft to +infty
    trans_imgs = img_transform(imgs) # (B,1,C,H,W)

    # initialized perturbation with a small standard deviation and the same size of input image batch
    mu = torch.randn_like(imgs) * 1e-3 # (B,1,C,H,W)

    for _ in range(N_step):
        # draw samples of gassian noise for each batch of images [N, C, H, W]
        samples = torch.randn((B, N, C, H, W), device=imgs.device, dtype=imgs.dtype) # [B, N, C, H, W]
        noise_update = mu + sigma * samples # [B,1,C,H,W] + sigma * [B, N, C, H, W]

        # candidate images (stay in [0,1])
        new_imgs = img_inverse_transform(trans_imgs + noise_update)  # (B,N,C,H,W)

        delta = new_imgs - imgs
        if att_type == 'infty':
            # enforce L_inf
            delta = torch.clamp(delta, -epsi, epsi)
            clip_imgs = imgs + delta
        elif att_type == 'l2':
            flat  = delta.flatten(2)                                             # [B,N,C*H*W]
            norms = flat.norm(p=2, dim=2, keepdim=True)                          # [B,N,1]
            scale = torch.clamp(norms / (epsi + 1e-12), min=1.0)
            delta = (flat / scale).view(B, N, C, H, W)
            clip_imgs = imgs + delta
        else:
            raise ValueError("att_type must be 'infty' or 'l2'")

        # evaluate batch
        logits = model(rearrange(clip_imgs, 'b n c h w -> (b n) c h w'))
        logits = rearrange(logits, '(b n) k -> b n k', b=B, n=N)

        # CW-style margin loss
        K = logits.shape[-1]
        labels = labels.view(B)  # ensure shape

        # expand labels across the N noise samples: [B, N]
        labels_exp = labels[:, None].expand(B, N)

        # true-class logit z_y: [B, N]
        real = logits.gather(2, labels_exp.unsqueeze(-1)).squeeze(-1) # .gather(2, [8,300,1])

        # max non-true logit max_{i≠y} z_i C&W like attack. It can be modified here to targeted attack!
        mask = F.one_hot(labels, num_classes=K).bool()          # [B, K]
        mask = mask[:, None, :].expand(B, N, K)                 # [B, N, K]
        other = logits.masked_fill(mask, float('-inf')).amax(dim=2)   # [B, N] other is the second largest label

        # "Higher = more correct" CW-style margin on logits:
        # the object is minimizing the margin to negative values 
        margin = (real - other).clamp_min(0.0)                  # [B, N]

        # If use the opposite as reward, since it is real - other(second largest)
        att_loss = reward_scaling * margin                                  # [B, N]
        z_score = (att_loss - att_loss.mean(dim=1, keepdim=True)) / (att_loss.std(dim=1, keepdim=True) + 1e-7)

        # ES grad per image: sum over noise dim
        # einsum version:
        grad_bchw = torch.einsum('bnchw,bn->bchw', samples, z_score)   # (B,C,H,W)

        # Update noise (keep noise as (B,1,C,H,W))
        mu = mu - (step_size / (N * sigma)) * grad_bchw.unsqueeze(1)
        ### Update the mu by NES end ###

        ### The interesting thing is the perturbation found is mu itself not the generated gaussian noise,
        ### the model was attacked by the mu...
 
    def out_fun():
        nonlocal mu
        ### Build final adversarials & outputs

        adv_imgs = img_inverse_transform(trans_imgs + mu).squeeze(1)                  # [B,C,H,W] (normalized)
        delta_final = adv_imgs - imgs.squeeze(1)                                        # [B,C,H,W] (normalized)

        # enforce epsilon one more time for safety
        if att_type == 'infty':
            delta_final = delta_final.clamp(-epsi, epsi)
        else:  # l2
            flat = delta_final.flatten(1)                                            # [B,C*H*W]
            norms = flat.norm(p=2, dim=1, keepdim=True)                              # [B,1]
            over = norms > (epsi + 1e-12)
            if over.any():
                scale = (epsi / (norms[over] + 1e-12))
                flat[over] = flat[over] * scale
                delta_final = flat.view(B, C, H, W)

        adv_imgs_clipped = imgs.squeeze(1) + delta_final

        # final preds on adversarials
        adv_logits = model(adv_imgs_clipped)
        adv_labels = adv_logits.argmax(dim=1)                                        # [B]
        success = (adv_labels != labels)                                             # BoolTensor [B]
        return success, adv_imgs_clipped, adv_labels

    success, adv_imgs, adv_labels = out_fun()

    return success, adv_imgs, adv_labels


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
    Attack a random fraction of samples from `dataloader`, but only those that are
    correctly classified. Record ONLY dataset indices where the attack succeeds.
    Expects each batch to include indices (either (imgs, labels, idxs) or dict with keys).
    Uses the batched attacker `n_attack_perbatch` (which returns: success, adv_imgs, delta, adv_labels).
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
    saved_perturbations: Optional[List[torch.Tensor]] = None
    if save_perturbations:
        saved_perturbations = []

    total_seen = 0           # total data points from all batches
    sampled = 0              # total data points sampled from all batches
    correct_on_clean = 0     # # correctly classified (eligible) across all batches
    attacked = 0             # # actually attacked (== correct_on_clean)
    successes = 0            # # successful attacks
    skipped_misclassified = 0  # sampled but misclassified on clean -> skipped

    for batch_i, batch in enumerate(dataloader):
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

        if keep_mask.sum().item() == 0:  # for tiny fractions, skip this batch if nothing kept
            continue

        # Subset by fraction
        imgs_keep = imgs[keep_mask]
        labels_keep = labels[keep_mask]
        idx_keep = idx_tensor[keep_mask]  # CPU

        # Move selected tensors to device
        imgs_keep = imgs_keep.to(device)
        labels_keep = labels_keep.to(device)
        imgs_keep = imgs_keep.to(device, non_blocking=True)
        labels_keep = labels_keep.to(device, non_blocking=True)

        # --- Filter to correctly classified (on clean) BEFORE attacking ---
        with torch.no_grad():
            clean_preds = model(imgs_keep).argmax(dim=1)

        correct_mask = (clean_preds == labels_keep)            # [B_k]
        num_correct = int(correct_mask.sum().item())
        correct_on_clean += num_correct
        skipped_misclassified += int(labels_keep.numel()) - num_correct

        if num_correct == 0:
            continue  # nothing to attack in this (fraction-filtered) subset

        imgs_corr   = imgs_keep[correct_mask]
        labels_corr = labels_keep[correct_mask]
        idx_corr    = idx_keep.cpu()[correct_mask.cpu()]             # Ensure both are on CPU

        # ---- Run the batched attacker ----
        # n_attack_perbatch must return: success [B_c], adv_imgs [B_c,C,H,W], delta [B_c,C,H,W], adv_labels [B_c]
        success_vec, adv_imgs, adv_labels = n_attack_perbatch(
            model, imgs_corr, labels_corr, **kwargs
        )

        # Book-keeping
        attacked += int(imgs_corr.size(0))
        num_success = int(success_vec.sum().item())
        successes  += num_success

        # Map successes back to dataset indices
        succ_mask_cpu = success_vec.detach().cpu().bool()
        succ_indices  = idx_corr[succ_mask_cpu].tolist()
        succeeded_indices.extend(succ_indices)

        # Optionally store perturbations (only for successes)

        if save_perturbations:

            # Only save successful attacks
            adv_imgs_succ = adv_imgs[succ_mask_cpu]      # [num_success, C, H, W]
            orig_imgs_succ = imgs_corr[succ_mask_cpu]    # [num_success, C, H, W]
            labels_succ = labels_corr[succ_mask_cpu]     # [num_success]
            adv_labels_succ = adv_labels[succ_mask_cpu]  # [num_success]

            num_success = adv_imgs_succ.size(0)
            if num_success > 0:
                # For each successful attack, create a figure with 2 images: original and adversarial
                fig, axes = plt.subplots(num_success, 2, figsize=(4, 2 * num_success))
                if num_success == 1:
                    axes = [axes]  # Make iterable

                for i in range(num_success):
                    # Original image
                    img_orig = orig_imgs_succ[i].detach().cpu()
                    img_adv = adv_imgs_succ[i].detach().cpu()
                    # Unnormalize if needed (assuming images in [0,1] or normalized)
                    img_orig = img_orig.clamp(0, 1)
                    img_adv = img_adv.clamp(0, 1)

                    axes[i][0].imshow(img_orig.permute(1, 2, 0).numpy())
                    axes[i][0].set_title(f"True: {labels_succ[i].item()}")
                    axes[i][0].axis('off')

                    axes[i][1].imshow(img_adv.permute(1, 2, 0).numpy())
                    axes[i][1].set_title(f"Pred: {adv_labels_succ[i].item()}")
                    axes[i][1].axis('off')

                # Ensure output directory exists
                out_dir = "./log/figures/nattack_results/"
                os.makedirs(out_dir, exist_ok=True)
                plt.tight_layout()
                # Save figure with a unique filename (e.g., by indices)
                fig_filename = os.path.join(
                    out_dir,
                    f"nattack_{batch_i}.png"
                )
                fig.savefig(fig_filename)
                saved_perturbations.append(fig_filename)
                plt.close(fig)


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
    }
