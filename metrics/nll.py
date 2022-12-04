from scipy.stats import norm
import torch

# "3.3 Data scaling, reverse process decoder, and L0" in the paper
# + https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/losses.py#L50  
def compute_nll(x, mean, std):
    upper_bound_cdf = norm.cdf(x.numpy() + 1 / 255, loc=mean.numpy(), sclae=std.numpy())
    lower_bound_cdf = norm.cdf(x.numpy() - 1 / 255, loc=mean.numpy(), sclae=std.numpy())
    # for x == -1
    log_upper_bound_cdf = torch.log(torch.from_numpy(upper_bound_cdf).clamp(min=1e-12))
    # for x == 1
    log_one_minus_lower_bound_cdf = torch.log(torch.from_numpy(1 - lower_bound_cdf).clamp(min=1e-12))
    # for -1 < x < 1
    log_delta_cdf = torch.log(torch.from_numpy(upper_bound_cdf - lower_bound_cdf).clamp(min=1e-12))
    log_likelihood = torch.where(
        x < -0.999, # = -1 
        log_upper_bound_cdf,
        torch.where(
            x > 0.999, # == 1
            log_one_minus_lower_bound_cdf,
            log_delta_cdf
        )
    )
    return -log_likelihood