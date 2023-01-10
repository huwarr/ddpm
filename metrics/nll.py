from scipy.stats import norm
import torch
import numpy as np
from tqdm.auto import tqdm

from data.dataloader import get_dataloaders

# "3.3 Data scaling, reverse process decoder, and L0" in the paper
# + https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/losses.py#L50  
def discrete_log_likelihood(x, mean, std):
    upper_bound_cdf = norm.cdf(x + 1 / 255, loc=mean, scale=std)
    lower_bound_cdf = norm.cdf(x - 1 / 255, loc=mean, scale=std)
    # for x == -1
    log_upper_bound_cdf = torch.log(torch.from_numpy(upper_bound_cdf).clamp(min=1e-12))
    # for x == 1
    log_one_minus_lower_bound_cdf = torch.log(torch.from_numpy(1 - lower_bound_cdf).clamp(min=1e-12))
    # for -1 < x < 1
    log_delta_cdf = torch.log(torch.from_numpy(upper_bound_cdf - lower_bound_cdf).clamp(min=1e-12))
    log_likelihood = torch.where(
        torch.from_numpy(x) < -0.999, # == -1 
        log_upper_bound_cdf,
        torch.where(
            torch.from_numpy(x) > 0.999, # == 1
            log_one_minus_lower_bound_cdf,
            log_delta_cdf
        )
    )
    return log_likelihood

# https://github.com/baofff/Extended-Analytic-DPM/blob/92aa7688a222e9845f903c0f4b98a6609b43d2c7/core/func/functions.py#L121
def kl_between_normal(mu_0, var_0, mu_1, var_1):
    tensor = None
    for obj in (mu_0, var_0, mu_1, var_1):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None

    var_0, var_1 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (var_0, var_1)
    ]

    return 0.5 * (var_0 / var_1 + (mu_0 - mu_1).pow(2) / var_1 + var_1.log() - var_0.log() - 1.)

# https://github.com/baofff/Extended-Analytic-DPM/blob/92aa7688a222e9845f903c0f4b98a6609b43d2c7/core/diffusion/likelihood.py#L26 
def compute_nll(model, dataset_name, T=1000, is_train=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    beta_s = np.linspace(1e-4, 0.02, T)
    alpha_s = 1 - beta_s
    alpha_s_new = np.cumprod(alpha_s)

    nll = 0.

    train_loader, test_loader = get_dataloaders(dataset_name)
    if is_train:
        loader = train_loader
    else:
        loader = test_loader

    for batch, _ in tqdm(loader):
        nelbo = torch.zeros(batch.size(0))

        for t in range(T - 1, -1, -1):
            n_samples = batch.shape[0]
            t_s = torch.tensor([t]).expand(n_samples).numpy()
            noise_s = np.random.normal(loc=0.0, scale=1.0, size=batch.shape)
            alpha_s_cur = alpha_s_new[t_s]
            noised_inputs = np.expand_dims(alpha_s_cur ** (1/2), axis=(1, 2, 3)) * batch.numpy() + np.expand_dims((1 - alpha_s_cur) ** (1/2), axis=(1, 2, 3)) * noise_s
            
            with torch.no_grad():
                t_s = torch.from_numpy(t_s).to(device)
                noised_inputs = torch.from_numpy(noised_inputs).float().to(device)
                targets = torch.from_numpy(noise_s).float().to(device)
                outputs = model(noised_inputs, t_s)
            
            batch = batch.cpu()
            outputs = outputs.cpu()
            noised_inputs = noised_inputs.cpu()
            
            multiplier = (1 - alpha_s[t]) / ((1 - alpha_s_new[t]) ** (1/2))
            mu_p = (noised_inputs - multiplier * outputs) / (alpha_s[t] ** (1/2))
            
            #z = np.random.normal(loc=0.0, scale=1.0, size=batch.size) if t > 0 else np.ones(batch.size)
            var_p = (beta_s[t] ** (1/2)) * torch.ones(batch.size())

            if t != 0:
                mu_q = alpha_s_new[t - 1] ** (1/2) * beta_s[t] / (1 - alpha_s_new[t]) * batch + \
                    alpha_s_new[t] ** (1/2) * (1 - alpha_s_new[t - 1]) / (1 - alpha_s_new[t]) * noised_inputs
                
                var_q = (1 - alpha_s_new[t - 1]) * beta_s[t] / (1 - alpha_s_new[t]) * torch.ones(batch.size())
                
                term = kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
            else:
                term = -discrete_log_likelihood(batch.numpy(), mu_p.numpy(), var_p.numpy()).flatten(1).sum(1)
            nelbo += term

        nll += nelbo.sum().item()
    return nll / len(loader.dataset)