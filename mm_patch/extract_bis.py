"""
The :mod:`sklearn.feature_extraction.image` submodule gathers utilities to
extract features from images.
"""

# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Olivier Grisel
#          Vlad Niculae
# License: BSD 3 clause

from itertools import product
import numbers
import numpy as np
from scipy import sparse
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

# from ..utils import check_array, check_random_state
# from ..base import BaseEstimator

__all__ = ['extract_patches_2d', 'patch_filter']


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

##############################################################################
# From an image to a set of small image patches

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    global_indices_shape : array with amount of patches horizonatlly and vertically
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    # arr is of type imageio.core.util.Array. 
    # Its stride is a bit wider than the pixels (guess it is because metadata)
    local_strides = arr.strides

    # st is the extraction_step for each dimension. slice(None, None, st) = [::st]
    slices = [slice(None, None, st) for st in extraction_step]
    global_strides = arr[tuple(slices)].strides

    global_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    # print(f'global_indices_shape: {global_indices_shape}')

    shape = tuple(list(global_indices_shape) + list(patch_shape))
    # print(f'shape: {shape}')
    strides = tuple(list(global_strides) + list(local_strides))
    # print(f'stride: {strides}')

    # as strided generates different views into the same array
    patches = as_strided(arr, shape=shape, strides=strides)
    return patches, global_indices_shape


def patch_filter(patch, lower_frac = 1. , upper_frac = 1. ):
    if not 0 <= lower_frac <= 1:
        raise ValueError("lower_frac must be between 0 and 1")
    if not 0 <= upper_frac <= 1:
        raise ValueError("upper_frac must be between 0 and 1")

    size = patch.size
    max_val = np.iinfo(patch.dtype).max 
    min_val = 0
    upper_thres = max_val - max_val*0.01
    lower_thres = min_val + max_val*0.01
    upper_lim = np.sum(patch > upper_thres) < upper_frac*size
    lower_lim = np.sum(patch < lower_thres) < lower_frac*size
    return  lower_lim and upper_lim  

def extract_patches_2d(image, patch_size, max_patches=None, stride = 1, patch_filter=None, random_state=None ):
    """Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.
    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # check_array is a function implemented in scikit learn
    # image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    # print(image.shape)

    extracted_patches, global_indices_shape = extract_patches(image,
                                        patch_shape = (p_h, p_w, n_colors),
                                        extraction_step = stride)

    
    avail_patches = global_indices_shape[0]*global_indices_shape[1]
    n_patches = min(avail_patches, max_patches)
    # _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    
    rng = check_random_state(random_state)
    mat_idx_arr = rng.choice(avail_patches, size=avail_patches, replace=False)
    

    i_s = []
    j_s = []
    if patch_filter:
        ok_patches = 0
        for idx in mat_idx_arr:
            i_global = idx // global_indices_shape[1]
            j_global = idx % global_indices_shape[1]
            if patch_filter(extracted_patches[i_global, j_global], lower_frac = 0.002 , upper_frac = 0.5):
                i_s.append(i_global)
                j_s.append(j_global)
                ok_patches += 1
                if ok_patches == n_patches:
                    break
        else:
            n_patches = ok_patches            
    else:
        for idx in mat_idx_arr[:n_patches]:
            i_global = idx // global_indices_shape[1]
            j_global = idx % global_indices_shape[1]
            i_s.append(i_global)
            j_s.append(j_global)
    
    patches = extracted_patches[i_s, j_s, 0]

    # this reshape forces a copy of the patches.
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        # astype(np.int32) is a requirement to be compatible with pytorch
        return patches.reshape((n_patches, p_h, p_w)).astype(np.int32)
    else:
        return patches.astype(np.int32)

if __name__ == '__main__':

    rng = check_random_state(11)
    
    print(rng.choice(10,size=10, replace = False))

