import os

from train import train_func
from sample import sample_func
from model.ddpm import UNet
from metrics.fid_and_is import compute_fid_and_is
from metrics.nll import compute_nll

import torch
from torchvision.utils import make_grid, save_image
import wandb
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset to train on")
parser.add_argument("--n_steps", type=int, default=800000, help="number of steps to train for")
parser.add_argument("--use_wandb", type=bool, default=False, help="whether to use wandb")
parser.add_argument("--wandb_key", type=str, default="", help="wandb key to login")
parser.add_argument("--sample_during_training", type=bool, default=False, help="whether to generate samples after each training epoch")
parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")

args = parser.parse_args()

assert args.dataset == 'CIFAR10' or args.dataset == 'MNIST', "This dataset is unavaliable. Please, choose MNIST or CIFAR10"

assert os.path.exists('./weights-inception-2015-12-05-6726825d.pth'), "Download InceptionV3 first, from here: https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"

logging.basicConfig(level=logging.NOTSET)
handle = ""
logger = logging.getLogger(handle)

IN_CHANNELS = 3
HID_CHANNELS = 128
DROPOUT = 0.1
TOTAL_STEPS = 1000
SEED = 42

# Training
torch.manual_seed(SEED)
ddpm_model = UNet(in_channels=IN_CHANNELS, hid_chahhels=HID_CHANNELS, dropout=DROPOUT, T=TOTAL_STEPS)
if args.use_wandb:
    wandb.login(key=args.wandb_key)
    wandb.init(project='ddpm_cifar10')
num_params = sum([p.numel() for p in ddpm_model.parameters()])
logger.info('The model has {} parameters'.format(num_params))
logger.info('Start training, steps: {}...'.format(args.n_steps))
losses, ema = train_func(ddpm_model, args.dataset, n_steps=args.n_steps, use_wandb=args.use_wandb, sample_during_training=args.sample_during_training, SEED=SEED, T=TOTAL_STEPS)
logger.info('Finished training!')
logger.info('Loss on training set: {}'.format(losses[-1]))

# Save checkpoints
torch.save(ddpm_model.state_dict(), 'model.pt')
torch.save(ema.state_dict(), 'model_ema.pt')

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, _ = sample_func(ddpm_model, in_channels=3, n_samples=args.n_samples, use_wandb=args.use_wandb, disable_tqdm=False)
samples_ema, _ = sample_func(ema, in_channels=3, n_samples=args.n_samples, use_wandb=args.use_wandb, with_ema=True, disable_tqdm=False)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_cifar10_samples.jpg and generated_cifar10_samples_ema.jpg ...')
grid = make_grid(samples, nrow=5)
save_image(grid, fp='generated_cifar10_samples.jpg')
grid_ema = make_grid(samples_ema, nrow=5)
save_image(grid_ema, fp='generated_cifar10_samples_ema.jpg')
logger.info('Done!')


# Compute metrics
logger.info('Calculate Frechet Inception Distance and Inception Score...')
fid_and_is = compute_fid_and_is(ema, '50000_samples', False)
is_mean = fid_and_is['inception_score_mean']
is_std = fid_and_is['inception_score_std']
fid = fid_and_is['frechet_inception_distance']
logger.info('FID = {}'.format(fid))
logger.info('IS = {} ± {}'.format(is_mean, is_std))

logger.info('Calculate NLL on Test Set...')
nll_on_test_set = compute_nll(ema, args.dataset, T=TOTAL_STEPS, is_train=False)
logger.info('NLL Test = {}'.format(nll_on_test_set))

logger.info('------ SCRIPT FINISHED SUCCESSFULLY ------')