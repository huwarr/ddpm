from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(real_data, generated_samples):
    fid = FrechetInceptionDistance(reset_real_features=False)
    fid.update(real_data, real=True)
    fid.update(generated_samples, real=False)
    return fid.compute().item()