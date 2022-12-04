import numpy as np
import torch

class CustomToTensor:
    """
    Convert PIL Image or numpy array to torch tensor.
    Scale the input image to [-1.0, 1.0]
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        arr = np.asarray(pic)
        arr = arr.astype(np.float32) / 127.5 - 1
        return torch.from_numpy(arr).permute(2, 0, 1)