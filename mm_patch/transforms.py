import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, swap = False):
        assert isinstance(swap, bool)
        self.swap = swap

    def __call__(self, array):

        if self.swap:
            # swap color axis because
            # numpy patch: H x W x C
            # torch patch: C X H X W
            array = array.transpose((2, 0, 1))

        return torch.from_numpy(array.astype('float32'))


class GrayToRGB(object):
    """Convert from grayscale to RGB."""

    def __call__(self, tensor):
        # patch, target = sample['patch'], sample['target']

        tensor = tensor.repeat(3, 1, 1)    
        return tensor

class Scale(object):
    """Scale torch tensor between max_range and min_range"""

    def __init__(self, min_range = 0, max_range = 1):
        self.min_range = min_range
        self.max_range = max_range

    def __call__(self, tensor ):
        
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        tensor = ((tensor.float() - min_val) / (max_val - min_val))*(self.max_range - self.min_range) + self.min_range
        # patch = ((patch - min_val) / (max_val - min_val))*(self.max_range - self.min_range) + self.min_range


        return tensor


