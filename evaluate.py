import os

from sample import sample_func
from model.ddpm import UNet
from metrics.fid_and_is import compute_fid_and_is
from metrics.nll import compute_nll

from ema_pytorch import EMA
import torch
from torchvision.utils import make_grid, save_image
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset to evaluate on")
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
ema = EMA(ddpm_model, beta = 0.9999)

logger.info('Load model from checkpoint...')
ema.load_state_dict(torch.load('model_ema.pt'))
logger.info('Successfully loaded the model!')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ema = ema.to(device)

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, _ = sample_func(ema, in_channels=IN_CHANNELS, n_samples=args.n_samples, use_wandb=False, with_ema=True, disable_tqdm=False)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_cifar10_samples_eval.jpg...')
grid = make_grid(samples, nrow=5)
save_image(grid, fp='generated_cifar10_samples_eval.jpg')
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
