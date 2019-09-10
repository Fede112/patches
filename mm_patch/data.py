import numpy as np
import os
import imageio
import numbers
import pandas as pd
import pickle
from PIL import Image
import re

import torch
from torch.utils.data import Dataset

from . import extract_bis as extract

def _listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

class PatchesPd():
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    def __init__(self, input_path, label, crop_seq, patch_size = 224, stride = 1, max_patches = 100, seed = 10):
        if not isinstance(input_path,str):
            raise TypeError("input_path must be a string.")
        if not isinstance(label,str):
            raise TypeError("label must be a string.")
        
        self.input_path = input_path
        self.label = label
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.seed = seed
        
        full_image_ls = _listdir_fullpath(input_path)[:14]
        
        patches_ls = []
        for path in full_image_ls:
            img = imageio.imread(path)
            img = crop_seq(img)
            #TODO: make patches.patch_filter a class
            single_img_patches = extract.extract_patches_2d(img, (self.patch_size,self.patch_size), self.max_patches, self.stride, extract.patch_filter, self.seed)
            single_img_patches = np.split(single_img_patches, single_img_patches.shape[0], axis = 0)
            # patches_ls += single_img_patches
            # patches_ls += [patch.reshape(patch.shape[1],patch.shape[2]) for patch in single_img_patches]
            patches_ls += [np.squeeze(patch, axis=0) for patch in single_img_patches]
            
        self.data = pd.DataFrame({'patch': patches_ls, 'label': self.label})
        

        print(f'number of {self.label} patches: {len(self.data)}')
        print(f'shape of single {self.label} patch: {self.data["patch"][0].shape}')


    def sample(self, num):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: The return value. True for success, False otherwise.

        .. _PEP 484:
            https://www.python.org/dev/peps/pep-0484/

        """
        return self.data['patch'].sample(num)
    
    

class Patches():
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    def __init__(self, input_path, output_path, crop_seq, \
                patch_size = 224, stride = 1, max_patches = 100, seed = 10):
        
        if not isinstance(input_path,str):
            raise TypeError("input_path must be a string.")
        if not isinstance(output_path,str):
            raise TypeError("output_path must be a string.")
        # if not isinstance(label,str):
            # raise TypeError("label must be a string.")
        
        self.input_path = input_path
        self.output_path = output_path
        self.label = os.path.basename(output_path)
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.seed = seed
        
        # full_image_ls = input_path
        full_image_ls = _listdir_fullpath(input_path)
        full_image_ls = full_image_ls
        
        import matplotlib.pyplot as plt
        counter = 0
        for path in full_image_ls:
            img = imageio.imread(path)

            # flip image if it is right oriented
            # This features is too specific for the mammograms problem. 
            # It shouldn't be include in a general purpose library       
            if re.findall('_(R)_', path):
                img = np.flip(img, axis=1) # axis = 1 means horizontal flip
                

            if crop_seq:
                img = crop_seq(img)
            
            #TODO: make patches.patch_filter a class
            single_img_patches = extract.extract_patches_2d(img, (self.patch_size,self.patch_size), self.max_patches, self.stride, extract.patch_filter, self.seed)
            single_img_patches = np.split(single_img_patches, single_img_patches.shape[0], axis = 0)
            # patches_ls += single_img_patches
            # patches_ls += [patch.reshape(patch.shape[1],patch.shape[2]) for patch in single_img_patches]
            
            for patch in single_img_patches:
                filename = '{:06}.png'.format(counter)
                # print(filename)
                filename_path = os.path.join(output_path,filename)
                img = Image.fromarray(np.squeeze(patch.astype(np.uint16), axis=0))
                imageio.imwrite(filename_path, img)
                counter+=1        

        


              


class PatchesDataset(Dataset):
    """Patches dataset."""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not isinstance(data_path,str):
            raise TypeError("data_path must be a string.")
        self.data_path = data_path
        
        with open(data_path, 'rb') as file:
            self.data = pickle.load(file)
        
        self.transform = transform
        self.label = self.data['label']
        self.patches = self.data['patch']
        self.unique_labels = list(set(self.label))
        self.unique_labels_dict = dict( zip(self.unique_labels, range(len(self.unique_labels))) )
        if pd.api.types.is_string_dtype(self.data['label']):
            self.data['target']= self.data['label'].map(self.unique_labels_dict) 
        else:
            self.data['target'] = self.data['label']
        self.target = self.data['target']


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

#         img_name = os.path.join(self.data_path,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
        patch = self.patches.iloc[idx]
        target = self.target.iloc[idx]

        if self.transform:
            patch = self.transform(patch)

        sample = {'patch': patch, 'target': target}

        return sample