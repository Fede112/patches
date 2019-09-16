import numpy as np
import os

def image_from_index(loader, index_ls):
    """Given a list of indexes, returns the corresponding list of images from the dataloader.

	The dataloader should not shuffle the images for this function to work well.

    
    Args:
        loader (dataloader): A pytorch dataloader without any shuffle enabled.
        index_ls (int): List of indexes of the images you want to obtain.

    Returns:
        list: Returns a list of images. The order of the images is the same as the one
        		specified by index_ls.

    """
    i = 0
    image_ls = [[]]*len(index_ls)
    for batch in loader:
        patches_batch, targets_batch = batch
        num_img = len(patches_batch)

        for idx in range(num_img):
            if i in index_ls:
                ls_idx = np.where(i==np.array(index_ls))[0][0]
                image_ls[ls_idx] = patches_batch[idx][0]
            i += 1
            
        # avoid to loop threw all dataset if images are already found
        if i>np.max(index_ls):
            break
    return image_ls