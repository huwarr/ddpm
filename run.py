from train import train_func
from sample import sample_func
from model.ddpm import UNet

import torch
from torchvision.utils import make_grid, save_image
import wandb
from PIL import Image
import argparse
import logging
import math


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset to train on")
parser.add_argument("--n_steps", type=int, default=800000, help="number of steps to train for")
parser.add_argument("--use_wandb", type=bool, default=False, help="whether to use wandb")
parser.add_argument("--wandb_key", type=str, default="", help="wandb key to login")
parser.add_argument("--sample_during_training", type=bool, default=False, help="whether to generate samples after each training epoch")
parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")

args = parser.parse_args()

assert args.dataset == 'CIFAR10' or args.dataset == 'MNIST', "This dataset is unavaliable. Please, choose MNIST or CIFAR10"

logging.basicConfig(level=logging.NOTSET)
handle = ""
logger = logging.getLogger(handle)

# Training
torch.manual_seed(42)
ddpm_model = UNet(in_channels=3, dropout=0.1, T=1000)
if args.use_wandb:
    wandb.login(key=args.wandb_key)
    wandb.init(project='ddpm_cifar10')
logger.info('Start training, steps: {}...'.format(args.n_steps))
losses = train_func(ddpm_model, args.dataset, n_steps=args.n_steps, use_wandb=args.use_wandb, sample_during_training=args.sample_during_training)
logger.info('Finished training!')
logger.info('Loss on training set: {}'.format(losses[-1]))

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, steps = sample_func(ddpm_model, n_samples=args.n_samples, use_wandb=args.use_wandb)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_cifar10_samples.jpg...')
grid = make_grid(samples, nrow=5)
save_image(grid, fp='generated_cifar10_samples.jpg')
logger.info('Done!')
