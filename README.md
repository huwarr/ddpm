# Implementation of "Denoising Diffusion Probabilistic Models" paper

This is a PyTorch implementation of the paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239).

## Training

First, install the dependencies by running:

`pip install -r requirements.txt`

The model can be trained via running `run.py` script. For example:

`python run.py --n_epochs 100`

The script will train the model, generate a couple of samples (by default 10, may be specified with `--n_samples` parameter), and save these sample into `generated_samples.jpg`.

## Evaluation with checkpoint

You may load a checkpoint of the trained model from the further discussed experiment, and generate samples with it with running:

`sh load_and_eval.sh`

This script will download a chekpoint `ddpm_trained.pt` and run `evaluate.py` script, which will generate 10 samples using a model, loaded from this chekpoint. Samples will be saved into `generated_samples.jpg`.

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

### Notebook

You may view code for this experiment in `train_model.ipynb`

## Repository structure

Main part:

- `model.py` - UNet definition

- `dataloader.py` - downloading MNIST dataset and creating dataloaders

- `train.py` - contains training function

- `sample.py` - contains function, that generates samples

Helpful scripts:

- `run.py` - a script to run training process and generate samples with trained model

- `evaluate.py` - a script to load model from checkpoint and generate samples

- `load_and_eval.sh` - a script to download chekpoint and run `evaluate.py`

Others:

- `train_model.ipynb` - a notebook with an above discussed experiment

- `pics/` - a folder with images to display in README (here)

- `generated_samples.jpg` - an example of what `run.py` and `evaluate.py` generate

- `requirements.txt` - necessary dependencies


## Sources

1. [[ARXIV] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

2. [[ARXIV] PixelCNN++](https://arxiv.org/abs/1701.05517)

3. [[ARXIV] Wide Residual Networks](https://arxiv.org/abs/1605.07146)
