from sample import sample_func
from model import UNet

import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import argparse
import logging
import math

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")

args = parser.parse_args()

logging.basicConfig(level=logging.NOTSET)
handle = ""
logger = logging.getLogger(handle)


ddpm = UNet()
logger.info('Load model from checkpoint...')
ddpm.load_state_dict(torch.load('ddpm_trained.pt'))
logger.info('Successfully loaded the model!')

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, steps = sample_func(ddpm, n_samples=args.n_samples)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_samples.jpg...')
grid = make_grid(samples, nrow=5)
save_image(grid, fp='generated_samples.jpg')
logger.info('Done!')
