from dataloader import get_dataloaders
from model import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    all_samples = [x]
    if use_wandb:
        wandb.log({"reverse step": wandb.Image(x[0])})
    
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
            all_samples.append(x)
            if use_wandb:
                wandb.log({"reverse step": wandb.Image(x[0])})
        
    if use_wandb:
        for sample in x:
            wandb.log({"samples": wandb.Image(sample)})

    return all_samples