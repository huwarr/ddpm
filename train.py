from dataloader import get_dataloaders
from model import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
import wandb

from sample import sample_func


SEED = 42

def train_func(model, n_epochs=5, use_wandb=False, sample_during_training=False):
    # Fix seed for reprodicibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    # Get train dataloader
    train_loader, _ = get_dataloaders()
    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Total number of steps
    T = 1000
    # Optimizer; they use this one with this LR in DDPM paper
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    # Loss - MSE
    criterion = nn.MSELoss()
    # ~Alpha_t for nosing the image for a particular number of steps in one go
    alpha_s_new = np.cumprod(1 - np.linspace(1e-4, 0.02, T))

    # Move the model to device and to a train mode
    model = model.to(device)
    model.train()

    # Training
    train_losses = []
    step = 0
    for n in range(n_epochs):
        for batch, _ in tqdm(train_loader, desc='Training, epoch {}'.format(n + 1)):
            # Get timestamps from unifor distribution
            n_samples = batch.shape[0]
            t_s = np.random.randint(low=0, high=T, size=n_samples)
            # Noise the images
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
        if sample_during_training:
            sample_func(model, use_wandb=use_wandb)
    
    return train_losses