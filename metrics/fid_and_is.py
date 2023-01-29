import os

import torch
import torch_fidelity
from torchvision.utils import save_image
from tqdm.auto import tqdm

from sample import sample_func

def compute_fid_and_is(model, out_dir, if_verbose=True):
    if len(os.listdir(out_dir)) < 50000:
        # Generate 50_000 samples
        n_samples_per_step = 500
        assert 50_000 % n_samples_per_step == 0
        counter = 0
        for _ in tqdm(range(50_000 // n_samples_per_step)):
            samples, _ = sample_func(model, n_samples=n_samples_per_step, use_wandb=False, SEED=None)
            for sample in samples:
                sample = (sample + 1) / 2
                save_image(sample, fp=os.path.join(out_dir, '{}.png'.format(counter)))
                counter += 1
    # Compute metrics
    # Output format:
    #   {'inception_score_mean': ...,
    #   'inception_score_std': ...,
    #   'frechet_inception_distance': ...}
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=out_dir, 
        input2='cifar10-train',
        batch_size=128,
        cuda=torch.cuda.is_available(),
        feature_extractor_weights_path='./weights-inception-2015-12-05-6726825d.pth',
        isc=True, 
        fid=True, 
        kid=False, 
        verbose=if_verbose
    )

    return metrics_dict
