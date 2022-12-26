from data.dataloader import get_dataloaders
from model.ddpm import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid, save_image

from tqdm.auto import tqdm
import wandb

from ema_pytorch import EMA

from sample import sample_func


SEED = 42

def train_func(
    model, dataset_name, n_steps=800_000, use_wandb=False, sample_during_training=False, sample_step=10000, SEED=42, T=1000, warmup=5000, grad_clip=1.0
):
    # Fix seed for reprodicibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    ema = EMA(
        model,
        beta = 0.9999
    )

    # Get train dataloader
    train_loader, _ = get_dataloaders(dataset_name)
    train_iter = iter(train_loader)
    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Total number of steps
    # Optimizer; they use this one with this LR in DDPM paper
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    # Loss - MSE
    criterion = nn.MSELoss()
    # ~Alpha_t for nosing the image for a particular number of steps in one go
    alpha_s_new = np.cumprod(1 - np.linspace(1e-4, 0.02, T))

    # Move the model to device and to a train mode
    model = model.to(device)
    model.train()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step, warmup) / warmup)

    # Training
    train_losses = []
    for n in tqdm(range(n_steps), desc='Training, step'):
        try:
            batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, _ = next(train_iter)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update()

        if use_wandb:
            wandb.log({'train loss': loss.item()}, step=n)

        train_losses.append(loss.item())
        if sample_during_training and n % sample_step == 0:
            samples, _ = sample_func(model, use_wandb=use_wandb)
            grid = make_grid(samples, nrow=5)
            save_image(grid, fp='samples/{}.png'.format(n))
            samples_ema, _ = sample_func(ema, use_wandb=use_wandb, with_ema=True)
            grid = make_grid(samples_ema, nrow=5)
            save_image(grid, fp='samples_ema/{}.png'.format(n))

    return train_losses, ema