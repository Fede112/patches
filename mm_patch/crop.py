# ==============================================================================
# This code is based on the image processing of the breast_cancer_classifier free software (Copyright (C) 2019 Nan Wu et al.): 
# you can redistribute it and/or modify it under the terms of the GNU Affero 
# General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or (at your option) any later version.
# ==============================================================================

import scipy.ndimage
import imageio
import numpy as np
import numbers
import matplotlib.pyplot as plt


# from . import extract

def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels+1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)
        
    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    # largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask_index = max(mask_pixels_dict, key=mask_pixels_dict.get)
    largest_mask = mask == largest_mask_index
    # F: full mask of largest component
    return largest_mask



def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    # returns True for rows, axis="y", or columns, axis="x", with non zero values
    # F: the np.arange is the index array which is sliced by has_value and then pick first and last element
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    # F: the plus one at the end maybe a source of error
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end

def crop_img_from_largest_connected(img, erode_dialate=True, iterations=100):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component, and draws a bounding box for the result
    with buffers

    input:
        - img:   2D numpy array
        - mode:  breast pointing left or right

    """
    # assert mode in ("left", "right")

    img_mask = img > 0

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)
    
    # figure out where to crop
    top, bottom = get_edge_values(img, largest_mask, "y")
    left, right = get_edge_values(img, largest_mask, "x")

    
    return img[top:bottom, left:right]


#TODO: transform into class
def crop_vertical(img, frac=[0.25, 0.01]):
    """
    Crop a fraction of the image at the top and bottom.
    If frac = 0.05 then you are removing 10% (0.05 top + 0.05 bot) of the vertical size.
    """

    img_ndim = img.ndim
    if isinstance(frac, numbers.Number):
        frac = tuple([frac] * img_ndim)
    
    rm_len = [int(img.shape[0]*f) for f in frac]
    return img[rm_len[0]:-rm_len[1],:]

#TODO: transform into class
def crop_horizontal(img, frac = [0.1,0.1]):
    """
    Crop a fraction of the image left and right.
    If frac = 0.05 then you are removing 10% (0.05 left + 0.05 right) of the horizontal size.
    """

    img_ndim = img.ndim
    if isinstance(frac, numbers.Number):
        frac = tuple([frac] * img_ndim)
    
    rm_len = [int(img.shape[1]*f) for f in frac]
    return img[:,rm_len[0]:-rm_len[1]]



class Crop():
    def __init__(self, transform_ls):
        self.transform_ls = transform_ls

    def __call__(self, img):
        for trans in self.transform_ls:
            img = trans(img)
        return img

if __name__ == '__main__':

    img = imageio.imread('/scratch/fbarone/dicom_CRO_23072019/sample_data/images/97800_L_CC.png')
    plt.imshow(img)
    plt.show()
    # img = crop_img_from_largest_connected(img)

    trans_ls = [crop_img_from_largest_connected, crop_horizontal, crop_vertical]
    crop_trans = Crop(trans_ls)
    img = crop_trans(img)
    plt.imshow(img)
    plt.show()
    # patches = mmpatch.extract_mmpatches_2d(img, (100, 100), 50, 10, mmpatch.patch_filter) 
    # print(patches.shape)