# Implementation of "Denoising Diffusion Probabilistic Models" paper

This is a PyTorch implementation of the paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239).

## Training

The model can be trained via running `run.py` script. For example:

`python run.py --n_epochs 100`

The script will train the model, generate a couple of samples (by default 10, may be specified with `--n_samples` parameter), and save these sample into `generated_samples.jpg`.

## Experiments

The model was trained for approximately 46k steps, which took 4 and a half hours with GPU 1/8 A100. We trained the model with Adam optimizer and learning rate 2e-4 (as in the paper).

This way we achieved `loss = 0.004` with the following dynamic:

<img src="pics/train loss.png" width="500" />

This number of training steps is clearly not enough, but we chose this nice trade off between training time and quality of generated samples because of limited resources.

### Generated samples:

<img src="pics/generated samples.png" width="400"/>

We can see, that the quality of samples is yet to be improved. However, every digit here is easy to recognize, which is a good sign.

Probably, we could have achieved better quality after the same amount of training time by using guidance with labels.

### Reverse diffusion process:

Here is an example of how the model denoises an image:

<img src="pics/reverse process.png" width="500"/>

## Sources

1. [[ARXIV] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

2. [[ARXIV] PixelCNN++](https://arxiv.org/abs/1701.05517)

3. [[ARXIV] Wide Residual Networks](https://arxiv.org/abs/1605.07146)
