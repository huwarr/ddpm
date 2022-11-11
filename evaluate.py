from train import train_func
from sample import sample_func
from model import UNet

import torch
from PIL import Image
import logging
import math

logging.basicConfig(level=logging.NOTSET)
handle = ""
logger = logging.getLogger(handle)


ddpm = UNet()
logger.info('Load model from checkpoint...')
ddpm.load_state_dict(torch.load('ddpm_trained.pt'))
logger.info('Successfully loaded the model!')

# Generate samples
logger.info('Generating {} samples...'.format(args.n_samples))
samples, steps = sample_func(ddpm, n_samples=args.n_samples, use_wandb=args.use_wandb)
logger.info('Finished generating!')

# Save samples
logger.info('Save generated samples to generated_samples.jpg...')
samples_PIL = [Image.fromarray(sample.squeeze().numpy()) for sample in samples]
rows = 2
cols = int(math.ceil(args.n_samples / rows))
# Python PIL/Image make 3x3 Grid from sequence Images: https://stackoverflow.com/a/65583584 
w, h = samples_PIL[0].size
grid = Image.new(mode='gray', size=(cols * w, rows * h))
grid_w, grid_h = grid.size
for i, img in enumerate(samples_PIL):
    grid.paste(img, box=(i%cols*w, i//cols*h))
grid.save("generated_samples.jpg")
logger.info('Done!')
