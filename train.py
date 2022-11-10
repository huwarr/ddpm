from dataloader import get_dataloaders
from model import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
import wandb



SEED = 42

def train_func(model, n_epochs=5, use_wandb=False, wandb_key=''):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    
    if use_wandb:
        wandb.login(key=wandb_key)
        wandb.init(project='ddpm')

    train_loader, _ = get_dataloaders()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    T = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    alpha_s_new = np.cumprod(1 - np.linspace(1e-4, 0.02, T))

    model = model.to(device)
    model.train()

    train_losses = []
    step = 0
    for n in range(n_epochs):
        for batch, _ in tqdm(train_loader, desc='Training, epoch {}'.format(n + 1)):
            n_samples = batch.shape[0]
            t_s = np.random.randint(low=0, high=T, size=n_samples)
            noise_s = np.random.normal(loc=0.0, scale=1.0, size=batch.shape)
            alpha_s_cur = alpha_s_new[t_s]
            noised_inputs = np.expand_dims(alpha_s_cur ** (1/2), axis=(1, 2, 3)) * batch.numpy() + np.expand_dims((1 - alpha_s_cur) ** (1/2), axis=(1, 2, 3)) * noise_s

            t_s = torch.from_numpy(t_s).to(device)
            noised_inputs = torch.from_numpy(noised_inputs).float().to(device)
            targets = torch.from_numpy(noise_s).float().to(device)

            optimizer.zero_grad()
            outputs = model(noised_inputs, t_s)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({'train loss': loss.item()}, step=step)
            step += 1
            train_losses.append(loss.item())
    
    return train_losses