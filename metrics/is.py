from torchmetrics.image.inception import InceptionScore


def compute_is(generated_samples):
    inception = InceptionScore()
    # generate some images
    inception.update(generated_samples)
    return inception.compute().item()