import matplotlib.pyplot as plt
import random


import mm_patch.data
from mm_patch.crop import *
import mm_patch.extract_bis as extract

def _read_image_png(file_name):
    image = np.array(imageio.imread(file_name)).astype(np.int32)
    return image


img = imageio.imread('/scratch/fbarone/images_full_patches_CRO_23072019/dense/117186_L_CC.png')

# define crop object
trans_ls = [crop_img_from_largest_connected, crop_horizontal, crop_vertical]
crop_seq = Crop(trans_ls)
img = crop_seq(img)

single_img_patches = extract.extract_patches_2d(img, (256,256), max_patches = 10, stride = 20, patch_filter = extract.patch_filter, random_state = 42)


# plt.imshow(single_img_patches[0])
# plt.show()


# mm_patch.data.Patches: build patches and save them as .png 
try:
    mm_patch.data.Patches('/scratch/fbarone/images_full_patches_CRO_23072019/dense', '/scratch/fbarone/patches_images_400/dense', crop_seq, patch_shape = 400, stride = 100, max_patches = 100)
    mm_patch.data.Patches('/scratch/fbarone/images_full_patches_CRO_23072019/venous', '/scratch/fbarone/patches_images_400/venous', crop_seq, patch_shape = 400, stride = 100, max_patches = 100)
except IOError as e:
    print(str(e))



random.seed(9)
shift_ds = mm_patch.data.DatasetShift('/scratch/fbarone/patches_images/', patch_shape = 100, shift = 50, loader = _read_image_png, extensions = 'png')

shift_ds[15]

