from dataloader import get_dataloaders
from model import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid

from tqdm.auto import tqdm
import wandb



SEED = 42

def sample_func(model, n_samples=10, use_wandb=False):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    
    #if use_wandb:
        #wandb.login(key=wandb_key)
        #wandb.init(project='ddpm')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    T = 1000
    beta_s = np.linspace(1e-4, 0.02, T)
    alpha_s = 1 - beta_s
    alpha_s_new = np.cumprod(alpha_s)

    model = model.to(device)
    model.eval()

    size = (n_samples, 1, 32, 32)
    x = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=size))
    reverse_steps = [x[0].tolist()]
    
    for t in tqdm(range(T - 1, -1, -1), desc='Sampling'):
        z = np.random.normal(loc=0.0, scale=1.0, size=size) if t > 0 else np.zeros(size)

        t_s = torch.tensor([t] * n_samples).to(device)
        x = x.to(device)
        with torch.no_grad():
            noise_predicted = model(x.float(), t_s)
        x = x.cpu()
        noise_predicted = noise_predicted.cpu()

        multiplier = (1 - alpha_s[t]) / ((1 - alpha_s_new[t]) ** (1/2))
        x = (x - multiplier * noise_predicted) / (alpha_s[t] ** (1/2)) + (beta_s[t] ** (1/2)) * torch.from_numpy(z)

        if t % 200 == 0:
            reverse_steps.append(x[0].tolist())
        
    if use_wandb:
        wandb.log({"samples": wandb.Image(make_grid(x, nrow=5))})
        wandb.log({"reverse step": wandb.Image(make_grid(torch.tensor(reverse_steps), nrow=len(reverse_steps)))})

    return x, torch.tensor(reverse_steps)