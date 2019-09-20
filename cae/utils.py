import numpy as np
import imageio

# based on https://github.com/foamliu/Autoencoder/blob/master/utils.py
class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.val = 0
        self.avg = 0
        self.count = 0
        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.val + (1 - self.alpha) * self.avg


def read_image_png(file_name):
    """
    Auxiliary loader function to create the dataset using 
    """
    image = np.array(imageio.imread(file_name)).astype(np.int32)
    return image