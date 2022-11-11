from train import train_func
from sample import sample_func
from model import UNet

import torch
from torchvision.utils import make_grid, save_image
import wandb
from PIL import Image
import argparse
import logging
import math


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--use_wandb", type=bool, default=False, help="whether to use wandb")
parser.add_argument("--wandb_key", type=str, default="", help="wandb key to login")
parser.add_argument("--sample_during_training", type=bool, default=False, help="whether to generate samples after each training epoch")
parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")

args = parser.parse_args()

logging.basicConfig(level=logging.NOTSET)
handle = ""
logger = logging.getLogger(handle)

# Training
torch.manual_seed(42)
ddpm = UNet(dropout=0.1)
if args.use_wandb:
    wandb.login(key=args.wandb_key)
    wandb.init(project='ddpm')
logger.info('Start training, epochs: {}...'.format(args.n_epochs))
losses = train_func(ddpm, n_epochs=args.n_epochs, use_wandb=args.use_wandb, sample_during_training=args.sample_during_training)
logger.info('Finished training!')
logger.info('Loss on training set: {}'.format(losses[-1]))

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, steps = sample_func(ddpm, n_samples=args.n_samples, use_wandb=args.use_wandb)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_samples.jpg...')
grid = make_grid(samples, nrow=5)
save_image(grid, fp='generated_samples.jpg')
logger.info('Done!')
