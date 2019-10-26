import sys
sys.path.insert(0,'/u/f/fbarone/Documents/patches/')

import torch
from models import Average_CAE_deep
from models import Average_CAE_deep_PCA


# import scipy.ndimage
# import numbers
import imageio
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


from mm_patch.crop import *
import mm_patch.transforms


img = imageio.imread('/scratch/fbarone/dicom_CRO_23072019/sample_data/images/97800_L_CC.png')
# plt.imshow(img)
# plt.show()

def scale_img(img, min_range = 0, max_range = 1):
    min_val = np.min(img)
    max_val = np.max(img)

    img = ((img.astype(float) - min_val) / (max_val - min_val))*(max_range - min_range) + min_range
    return img


trans_ls = [crop_img_from_largest_connected, scale_img]
crop_trans = Crop(trans_ls)
trans_img = crop_trans(img)
# plt.imshow(trans_img)
# plt.show()

print(trans_img.shape)

px = 256
py = 256
num_prow = (trans_img.shape[0] // 256)
num_pcol = (trans_img.shape[1] // 256)

output_img_256x8x8 = np.zeros(trans_img.shape)
output_img_1024 = np.zeros(trans_img.shape)
output_img_100 = np.zeros(trans_img.shape)

# Transfer Learning
pretrained_weights = True


# 256x8x8
pretrained_model_256x8x8 = './output/cae_models/20191009-232545_Average_CAE_deep-256x8x8.pt' # general weights
#1024
pretrained_model_1024 = './output/cae_models/20191018-210808_Average_CAE_deep_PCA-1024.pt'
#100
pretrained_model_100 = './output/cae_models/20191022-192609_Average_CAE_deep_PCA-100-256x8x8.pt'


print("----------------------------------------------------------------")


# check for nvidia library
assert torch.has_cudnn == True, 'No cudnn library!'

# set device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# initialize the NN model
model_256x8x8 = Average_CAE_deep().to(device)
model_1024 = Average_CAE_deep_PCA(1024).to(device)
model_100 = Average_CAE_deep_PCA(100).to(device)

# the manual seed needs to be set after the model is initialized so as to have the same seed for the dataloader.
# the initialization functions change the state of the random seed. Different models change the seed in different ways.
torch.manual_seed(15)


# pretrained weights
pretrained_model_dict = {}
if pretrained_weights:
    print("----------------------------------------------------------------")
    print("Loading pretrained weights... ", end='')

    ## Load subset of pretrained model
    # sub_model keys
    model_256x8x8_dict = model_256x8x8.state_dict()
    model_1024_dict = model_1024.state_dict()
    model_100_dict = model_100.state_dict()
    # load full model pretrained dict
    pretrained_model_256x8x8_dict = torch.load(pretrained_model_256x8x8)['state_dict']
    pretrained_model_1024_dict = torch.load(pretrained_model_1024)['state_dict']
    pretrained_model_100_dict = torch.load(pretrained_model_100)['state_dict']
    # pretrained_pca = torch.load(pca_filepath)

    # From pytorch discuss: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16
    # 1. filter out unnecessary keys
    pretrained_model_256x8x8_dict = {k: v for k, v in pretrained_model_256x8x8_dict.items() if k in model_256x8x8_dict}
    pretrained_model_1024_dict = {k: v for k, v in pretrained_model_1024_dict.items() if k in model_1024_dict}
    pretrained_model_100_dict = {k: v for k, v in pretrained_model_100_dict.items() if k in model_100_dict}
    # pretrained_pca = {k: v for k, v in pretrained_pca.items() if k in model_dict}
    
    # 2. overwrite entries in the existing state dict
    model_256x8x8_dict.update(pretrained_model_256x8x8_dict) 
    model_1024_dict.update(pretrained_model_1024_dict) 
    model_100_dict.update(pretrained_model_100_dict) 
    # model_dict.update(pretrained_pca) 
    # 3. load the new state dict
    model_256x8x8.load_state_dict(model_256x8x8_dict)
    model_1024.load_state_dict(model_1024_dict)
    model_100.load_state_dict(model_100_dict)


    print("Done!")

    print("----------------------------------------------------------------")


# get sample outputs
model_256x8x8.eval()
model_1024.eval()
model_100.eval()



for block_i in range(num_prow):
	for block_j in range(num_pcol):
		patch = torch.tensor(trans_img[block_i*px:block_i*px + px , block_j*py:block_j*py + py])
		# print(patch.shape)
		patch = patch.unsqueeze(0).unsqueeze(0)
		# print(patch.shape)
		output_256x8x8 = model_256x8x8(patch.float().to(device=device))[0,:,:]
		output_256x8x8 = output_256x8x8.detach().cpu().numpy()
		output_1024 = model_1024(patch.float().to(device=device))[0,:,:]
		output_1024 = output_1024.detach().cpu().numpy()
		output_100 = model_100(patch.float().to(device=device))[0,:,:]
		output_100 = output_100.detach().cpu().numpy()

		output_img_256x8x8[block_i*px:block_i*px + px , block_j*py:block_j*py + py] = output_256x8x8
		output_img_1024[block_i*px:block_i*px + px , block_j*py:block_j*py + py] = output_1024
		output_img_100[block_i*px:block_i*px + px , block_j*py:block_j*py + py] = output_100
		# output = model_256x8x8(patch.float().to(device=device))
# output_256x8x8 = model_256x8x8(images.to(device=device))
# output_1024 = model_1024(images.to(device=device))
# output_100 = model_100(images.to(device=device))

# prep images for display
# images = images.cpu().numpy()

fig = plt.figure(figsize=(40,10))
ax_input = fig.add_subplot(141)
ax_256x8x8 = fig.add_subplot(142)
ax_1024 = fig.add_subplot(143)
ax_100 = fig.add_subplot(144)
ax_input.imshow(trans_img)
ax_100.imshow(output_img_100)
ax_1024.imshow(output_img_1024)
ax_256x8x8.imshow(output_img_256x8x8)

ax_input.set_title('Input', fontsize = 12)
ax_256x8x8.set_title('CAE [CF=4]')
ax_1024.set_title('CAE+PCA-1024 [CF=64]')
ax_100.set_title('CAE+PCA-100 [CF=640]')

ax_input.get_xaxis().set_visible(False)
ax_100.get_xaxis().set_visible(False)
ax_1024.get_xaxis().set_visible(False)
ax_256x8x8.get_xaxis().set_visible(False)

ax_input.get_yaxis().set_visible(False)
ax_100.get_yaxis().set_visible(False)
ax_1024.get_yaxis().set_visible(False)
ax_256x8x8.get_yaxis().set_visible(False)


        # ax.get_xaxis().set_visible(False)


plt.show()