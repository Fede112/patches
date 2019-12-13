import numpy as np
import os, sys
import imageio
import random
import numbers
import pandas as pd
import pickle
from PIL import Image
import re

import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import has_file_allowed_extension


from . import extract_bis as extract


import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        patch_shape (int): Size of the square patch.
        stride (int): Stride between patches. It is the same in both directions.
        max_patches (int): Maximum amount of patches per image.
        seed (int): set a random seed to make the extraction pattern deterministic.

    Methods:
        sample(num): samples num patches from the dataframe.

    Returns:
        dataframe (pandas.DataFrame): A dataframe containing all the extracted patches.

    """
    def __init__(self, input_path, label, crop_seq, patch_shape = 224, stride = 1, max_patches = 100, seed = 10):
        if not isinstance(input_path,str):
            raise TypeError("input_path must be a string.")
        if not isinstance(label,str):
            raise TypeError("label must be a string.")

        self.input_path = input_path
        self.label = label
        self.patch_shape = patch_shape
        self.stride = stride
        self.max_patches = max_patches
        self.seed = seed
        
        full_image_ls = _listdir_fullpath(input_path)
        
        patches_ls = []
        for path in full_image_ls:
            img = imageio.imread(path)
            img = crop_seq(img)
            #TODO: make patches.patch_filter a class
            single_img_patches = extract.extract_patches_2d(img, (self.patch_shape,self.patch_shape), self.max_patches, self.stride, extract.patch_filter, self.seed)
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
        patch_shape (int): Size of the square patch.
        stride (int): Stride between patches. It is the same in both directions.
        max_patches (int): Maximum amount of patches per image.
        seed (int): set a random seed to make the extraction pattern deterministic.

    """
    def __init__(self, input_path, output_path, crop_seq, \
                patch_shape = 224, stride = 1, max_patches = 100, seed = 10):
        
        if not isinstance(input_path,str):
            raise TypeError("input_path must be a string.")
        if not isinstance(output_path,str):
            raise TypeError("output_path must be a string.")
        # if not isinstance(label,str):
            # raise TypeError("label must be a string.")
        

        if os.path.exists(output_path):
            # Prevent overwriting to an existing directory
            raise IOError(f"The directory '{output_path}' to save the patches already exists.")
        else:
            os.makedirs(output_path)
        

        self.input_path = input_path
        self.output_path = output_path
        self.label = os.path.basename(output_path)
        self.patch_shape = patch_shape
        self.stride = stride
        self.max_patches = max_patches
        self.seed = seed
        
        # full_image_ls = input_path
        full_image_ls = _listdir_fullpath(input_path)
        full_image_ls = full_image_ls[:10]
        
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
            single_img_patches = extract.extract_patches_2d(img, (self.patch_shape,self.patch_shape), self.max_patches, self.stride, extract.patch_filter, self.seed)
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





def make_dataset(dir, class_to_idx, patch_shape, shift, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for label in sorted(class_to_idx.keys()):
        d = os.path.join(dir, label)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    # x_shift = random.uniform(-shift,shift)
                    # sign = 1 if random.random() < 0.5 else -1
                    # y_shift = np.sqrt(shift*shift - x_shift*x_shift)*sign

                    # vertical axis
                    x_shift = 0
                    # horizontal axis
                    y_shift = shift

                    target_shift = (int(np.floor(x_shift)), int(np.floor(y_shift)))
                    # item = (path, target_shift, class_to_idx[label])
                    item = (path, target_shift)
                    images.append(item)

    return images



class DatasetShift(VisionDataset):
    """A data set where the samples are arranged in this way
    based on torchvision.datasets.folder.DatasetFolder: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, patch_shape=256, 
                    shift=None, extensions=None, transform=None, 
                    target_transform=None, is_valid_file=None):

        super(DatasetShift, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, patch_shape, shift, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        if isinstance(patch_shape, numbers.Number):
            patch_shape = tuple([patch_shape] * 2)

        if len(patch_shape) != 2:
            raise ValueError("'patch_shape' dim != 2. \n Dataset allows only 2d images.")

        # loader function
        self.loader = loader
        self.extensions = extensions

        

        self.patch_shape = patch_shape
        self.shift = shift

        self.classes = classes
        # dict with classes to idx
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, shift = self.samples[index]
        full_img = self.loader(path)

        x_center, y_center = np.array(full_img.shape)//2
        sample = full_img[x_center - self.patch_shape[0]//2 : x_center + self.patch_shape[0]//2,
                          y_center - self.patch_shape[1]//2 : y_center + self.patch_shape[1]//2]
        sample = sample[:,:, None]
        
        target = full_img[x_center - self.patch_shape[0]//2 + shift[0]: x_center + self.patch_shape[0]//2 + shift[0],
                          y_center - self.patch_shape[1]//2 + shift[1]: y_center + self.patch_shape[1]//2 + shift[1]]
        
        # Plot verifications (to be removed)
        # print(full_img.shape)
        # print(shift)
        # fig = plt.figure()
        # ax = fig.add_subplot(1,3,1)
        # ax1 = fig.add_subplot(1,3,2)
        # ax2 = fig.add_subplot(1,3,3)
        # ax.imshow(full_img, vmin = 0, vmax = np.max(full_img))
        # ax1.imshow(np.squeeze(sample), vmin = 0, vmax = np.max(full_img))
        # ax2.imshow(target, vmin = 0, vmax = np.max(full_img))
        # ax.set_title('Full image')
        # # Rectangles have xy inverted with respect to imshow
        # rect_patch = patches.Rectangle((y_center - self.patch_shape[1]//2, x_center - self.patch_shape[0]//2), self.patch_shape[1], self.patch_shape[0], fill = False)
        # rect_shift = patches.Rectangle((y_center - self.patch_shape[1]//2 + shift[1], x_center - self.patch_shape[0]//2 + shift[0]), self.patch_shape[1], self.patch_shape[0], fill = False, edgecolor='r')
        # ax.add_patch(rect_patch)
        # ax.add_patch(rect_shift)
        # ax1.set_title(f'center patch')
        # ax2.set_title(f'shifted patch: {shift}')
        # plt.show()


        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)

