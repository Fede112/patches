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
    """Returns a full path list for every file inside the input directory."""

    return [os.path.join(d, f) for f in os.listdir(d)]

class PatchesPd():
    """Creates a pandas dataframe with 2d patches randomly extracted from 2d full images.

    The full images should be png stored in a specific folder.
    The patches are randomly extracted from each image, but keeping the samples per image constant
    whenever this is possible.
    The stride parameters defines the overlap between the patches.

    Args:
        input_path (str): Path of the directory where the full images are stored.
        label (str): Label that describes the type of patches
        crop_seq (list): List of crop functions to be applied to the full images before extracting the patches.
        patch_size (int): Size of the square patch.
        stride (int): Stride between patches. It is the same in both directions.
        max_patches (int): Maximum amount of patches per image.
        seed (int): set a random seed to make the extraction pattern deterministic.

    Methods:
        sample(num): samples num patches from the dataframe.

    Returns:
        dataframe (pandas.DataFrame): A dataframe containing all the extracted patches.

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
        print(full_image_ls)
        
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
        """Samples a number of patches from the dataframe.

        Args:
            num (int): Number of patches to sample.

        Returns:
            Dataframe with the random samples
        """
        return self.data['patch'].sample(num)
    
    

class Patches():
    """Creates 2d patches randomly extracted from 2d full images.

    The full images should be png stored in a user input folder.
    The patches are stored in a user input folder.
    The patches are randomly extracted from each image, but keeping the samples per image constant
    whenever this is possible.
    The stride parameters defines the overlap between the patches.


    Args:
        input_path (str): Path of the directory where the full images are stored.
        output_path (str): Path of the directory where the patches will be saved.
        crop_seq (list): List of crop functions to be applied to the full images before extracting the patches.
        patch_size (int): Size of the square patch.
        stride (int): Stride between patches. It is the same in both directions.
        max_patches (int): Maximum amount of patches per image.
        seed (int): set a random seed to make the extraction pattern deterministic.

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
    """A Pytorch dataset from a PatchesPd instance."""

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