from data.dataloader import get_dataloaders
from model.ddpm import UNet
from metrics.nll import compute_nll

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid

from tqdm.auto import tqdm
import wandb



def sample_func(model, in_channels=3, n_samples=10, log_step=200, use_wandb=False, with_ema=False, SEED=42, disable_tqdm=True):
    # Fix seed to get the same pictures every time and see, how they are improving over time
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Total number of diffusion process steps
    T = 1000
    # The authors suggest 2 ways of defining sigma_t, one of them is sigma_t^2 = beta_t.
    # Let's stick to this one
    beta_s = np.linspace(1e-4, 0.02, T)
    # Apla_t and ~alpha_t for denosing an image
    alpha_s = 1 - beta_s
    alpha_s_new = np.cumprod(alpha_s)
    # Model
    model = model.to(device)
    model.eval()
    # Let's get pure Gaussian noise to start with in the reverse diffusion process
    size = (n_samples, in_channels, 32, 32)
    x = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=size))
    # Save an example of how a particular image is being denoised to visualize later
    reverse_steps = [x[0].tolist()]
    # Finally, reverse diffusion process
    for t in tqdm(range(T - 1, -1, -1), desc='Sampling', disable=disable_tqdm):
        z = np.random.normal(loc=0.0, scale=1.0, size=size) if t > 0 else np.zeros(size)

        t_s = torch.tensor([t] * n_samples).to(device)
        x = x.to(device)
        with torch.no_grad():
            noise_predicted = model(x.float(), t_s)
        x = x.cpu()
        noise_predicted = noise_predicted.cpu()

        multiplier = (1 - alpha_s[t]) / ((1 - alpha_s_new[t]) ** (1/2))
        mean = (x - multiplier * noise_predicted) / (alpha_s[t] ** (1/2))
        x = mean + (beta_s[t] ** (1/2)) * torch.from_numpy(z)

        if t % log_step == 0 or t == 0:
            reverse_steps.append(x[0].tolist())
    
    x = torch.clip(x, -1, 1)
    reverse_steps = torch.clip(torch.tensor(reverse_steps), -1, 1)

    x = (x + 1) / 2
    reverse_steps = (reverse_steps + 1) / 2
        
    if use_wandb:
        samples_title = 'samples (with EMA)' if with_ema else 'samples'
        reverse_step_title = 'reverse step (with EMA)' if with_ema else 'reverse step'
        wandb.log({samples_title: wandb.Image(make_grid(x, nrow=5))})
        wandb.log({reverse_step_title: wandb.Image(make_grid(reverse_steps, nrow=reverse_steps.shape[0]))})

    return x, reverse_steps