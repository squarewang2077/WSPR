import torch
import torch.nn.functional as F


def n_attack(model = None, img = None, ground_truth = None, N_norm = 300, \
             N_step = 500, step_size = 0.02, sigma = 0.1, norm_type = 'infty', epsi = 0.031, \
             reward_scaling = 0.5, eps = 1e-30):

    """    
    N_attack implement on a given model.
    Args:
        N_norm (int): Number of gaussian noise samples to use for the attack
        N_step (int): Number of steps to run the attack
        img (torch.Tensor): Input images to attack, shape (C, H, W)
    """

    def arctanh_torch(x: torch.Tensor) -> torch.Tensor:
    # stable arctanh on (-1, 1)
        return 0.5 * torch.log((1 + x + 1e-12) / (1 - x + 1e-12))

    def img_transform(img: torch.Tensor) -> torch.Tensor:
        """Transform the input image to tanh-space."""
        return arctanh_torch((img - 0.5) / (0.5 + 1e-12))

    def img_inverse_transform(img: torch.Tensor) -> torch.Tensor:
        """Inverse transform from tanh-space to [0,1]"""
        return torch.tanh(img) * 0.5 + 0.5  

    # extend the input image to a batch of size 1
    img = img.unsqueeze(0) # (1,C,H,W)

    # transform the input image to tanh-space of -inft to +infty
    trans_img = img_transform(img) # (1,C,H,W)

    # initialized perturbation with a small standard deviation and the same size of input image batch
    noise = torch.randn_like(img) * 1e-3 # (1,C,H,W)

    for _ in range(N_step):
        # draw samples of gassian noise for each batch of images [N_norm, C, H, W]
        samples = torch.randn((N_norm, *img.shape), device=img.device, dtype=img.dtype) # [N_norm, C, H, W]
        noise_update = noise.repeat(N_norm, 1, 1, 1) + sigma * samples # [N_norm, C, H, W] + sigma * samples

        # candidate images (stay in [0,1])
        new_imgs = img_inverse_transform(trans_img + noise_update)  # (N,C,H,W)

        delta = new_imgs - img
        if norm_type == 'infty':
            # enforce L_inf
            delta = torch.clamp(delta, -epsi, epsi)
            clip_imgs = img + delta
        elif norm_type == 'l2':
            # enforce L2 norm
            norm = torch.norm(delta.view(N_norm, -1), p=2, dim=1, keepdim=True)
            scale = torch.clamp(norm / epsi, min=1e-8)
            clip_imgs = img + delta / scale.view(N_norm, 1, 1)

        # evaluate batch
        logits = model(clip_imgs)  # (N,10)
        probs = F.softmax(logits, dim=-1)

        # CW-style margin loss
        target_onehot = torch.zeros((N_norm, 10), device=probs.device, dtype=probs.dtype)
        target_onehot[:, ground_truth] = 1.0

        real = torch.log((target_onehot * probs).sum(dim=1) + eps)
        other = torch.log((probs - target_onehot * 1e4).max(dim=1).values + eps)
        loss = torch.clamp(real - other, min=0.0, max=1e3)  # higher = more correct

        # ES gradient estimate
        reward = reward_scaling * loss
        A = (reward - reward.mean()) / (reward.std() + 1e-7)

        img_shape = img.squeeze(0).shape
        grad = (samples.view(N_norm, -1).T @ A).view(img_shape)
        noise += (step_size / (N_norm * sigma)) * grad





