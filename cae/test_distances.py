import sys
sys.path.insert(0,'/u/f/fbarone/Documents/patches/')

import torch
from torchvision import transforms
from tqdm import tqdm



from models import ModifiedDenseNet121
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
import mm_patch.extract_bis as extr


# from numpy import linalg as LA  
from scipy.spatial import distance


# img = imageio.imread('/u/f/fbarone/Desktop/mercury.png')
# img = img[:,:,0]
# print(img.shape)
# plt.imshow(img)
# plt.show()

def scale_img(img, min_range = 0, max_range = 1):
    min_val = np.min(img)
    max_val = np.max(img)

    img = ((img.astype(float) - min_val) / (max_val - min_val))*(max_range - min_range) + min_range
    return img


to_tensor = transforms.ToTensor()
densenet_norm = transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])




# CONFIGURATION

# Images list
images_ls = ['80907_L_CC', '97800_L_CC', '116588_L_CC', '117186_L_CC', '118343_L_CC', '121445_L_CC', '122118_L_CC', '122141_L_CC', '122876_L_CC', '132796_L_CC' ]
# images_ls = ['80907_L_CC', '97800_L_CC']
# Patches per image
patches_per_image = 40

# Transfer Learning
pretrained_weights = True












# densenet
pretrained_densenet_filepath = '/u/f/fbarone/Documents/breast_cancer_classifier/models/sample_patch_model.p'
# 256x8x8
pretrained_model_256x8x8 = './output/cae_models/20191009-232545_Average_CAE_deep-256x8x8.pt' # general weights
# 1024
pretrained_model_1024 = './output/cae_models/20191018-210808_Average_CAE_deep_PCA-1024.pt'
# 100
pretrained_model_100 = './output/cae_models/20191022-192609_Average_CAE_deep_PCA-100-256x8x8.pt'


print("----------------------------------------------------------------")


# check for nvidia library
assert torch.has_cudnn == True, 'No cudnn library!'

# set device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# initialize the NN model
dense_param = {'num_classes': 4}
# model = DenseNet_CAE_PCA(dense_param).to(device)
model_densenet = ModifiedDenseNet121(**dense_param).to(device)
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


    ## Load Densenet model
    model_densenet.load_from_path(pretrained_densenet_filepath)


    ## Load subset of pretrained model
    # sub_model keys
    model_256x8x8_dict = model_256x8x8.state_dict()
    model_1024_dict = model_1024.state_dict()
    model_100_dict = model_100.state_dict()
    # load full model pretrained dict
    pretrained_model_256x8x8_dict = torch.load(pretrained_model_256x8x8, map_location='cpu')['state_dict']
    pretrained_model_1024_dict = torch.load(pretrained_model_1024, map_location='cpu')['state_dict']
    pretrained_model_100_dict = torch.load(pretrained_model_100, map_location='cpu')['state_dict']
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
model_densenet.eval()


# px = 256
# py = 256
# num_prow = (trans_img.shape[0] // 256) #+ 1
# num_pcol = (trans_img.shape[1] // 256) #+ 1

# output_img_256x8x8 = np.zeros(trans_img.shape)
# output_img_1024 = np.zeros(trans_img.shape)
# output_img_100 = np.zeros(trans_img.shape)



trans_ls = [crop_img_from_largest_connected]

output_densenet_ls = []
output_256x8x8_ls = []
output_1024_ls = []
output_100_ls = []




for img_str in tqdm(images_ls): 
    
    img = imageio.imread('/scratch/fbarone/dicom_CRO_23072019/sample_data/images/'+img_str+'.png')
    crop_trans = Crop(trans_ls)
    trans_img = crop_trans(img)
    indexes, patches = extr.extract_patches_2d(trans_img, (256,256), max_patches=patches_per_image, stride=25, 
                                                patch_filter=extr.patch_filter, random_state=13)



    for patch in patches:
        # patch = torch.tensor(trans_img[block_i*px:block_i*px + px , block_j*py:block_j*py + py])
        # print(patch.shape)
        # patch = torch.tensor(patch)
        patch = scale_img(patch)
        patch = torch.from_numpy(patch.astype('float32'))
        patch = patch.unsqueeze(0).unsqueeze(0)

        # print(patch.shape)
        output_256x8x8 = model_256x8x8(patch.float().to(device=device))
        output_256x8x8 = output_256x8x8.unsqueeze(0).flatten()
        output_256x8x8 = output_256x8x8.detach().cpu().numpy()
        # print(output_256x8x8.shape)
        output_1024 = model_1024(patch.float().to(device=device))
        output_1024 = output_1024.unsqueeze(0).flatten()
        output_1024 = output_1024.detach().cpu().numpy()
        # print(output_1024.shape)
        output_100 = model_100(patch.float().to(device=device))
        output_100 = output_100.unsqueeze(0).flatten()
        output_100 = output_100.detach().cpu().numpy()


        # print(patch.shape)
        # patch = to_tensor(patch)
        patch = patch[0,:,:,:]
        patch = patch.repeat(3, 1, 1)
        patch = densenet_norm(patch)
        patch.unsqueeze_(0)
        # print(patch.shape)

        output_densenet = model_densenet(patch.float().to(device=device))
        output_densenet = output_densenet.unsqueeze(0).flatten()
        # print(output_densenet.shape)
        output_densenet = output_densenet.detach().cpu().numpy()
        

        output_densenet_ls.append(output_densenet)
        output_256x8x8_ls.append(output_256x8x8)
        output_1024_ls.append(output_1024)
        output_100_ls.append(output_100)

# Figure 1024 against Densenet

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, len(images_ls)*patches_per_image, patches_per_image)
# minor_ticks = np.arange(0, 401, 5)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='major', alpha=1, c = 'black')
# ax.grid(which='minor', alpha=0.2)

        
# print(len(output_1024_ls))
distance_matrix = np.zeros( (len(output_1024_ls),len(output_1024_ls)) )
for i in range(len(output_1024_ls)):
    for j in range(i,len(output_1024_ls)):
        dist = distance.euclidean(output_1024_ls[i], output_1024_ls[j])
        print(f'CAE_1024: {dist}')
        distance_matrix[i,j] = dist
        distance_matrix[j,i] = distance_matrix[i,j]

ax.imshow(distance_matrix)
ax.set_title('CAE-PCA-1024 features')
ax.grid(True)







pca_trans_path = '/u/f/fbarone/Documents/patches_analysis/73000/densenet_pca_0.95_eigenvectors_256x256.txt'

pca_trans = np.loadtxt(pca_trans_path)

print(pca_trans.shape)

ax_d = fig.add_subplot(1, 2, 2)

ax_d.set_xticks(major_ticks)
# ax_d.set_xticks(minor_ticks, minor=True)
ax_d.set_yticks(major_ticks)
# ax_d.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax_d.grid(which='both')

# Or if you want different settings for the grids:
ax_d.grid(which='major', alpha=1, c = 'black')
# ax_d.grid(which='minor', alpha=0.2)



distance_matrix = np.zeros( (len(output_densenet_ls),len(output_densenet_ls)) )
for i in range(len(output_densenet_ls)):
    for j in range(i,len(output_densenet_ls)):
        # print(np.dot(pca_trans,output_densenet_ls[i]).shape)

        dist = distance.euclidean(  np.dot(pca_trans,output_densenet_ls[i]), np.dot(pca_trans,output_densenet_ls[j])   )
        print(dist)
        distance_matrix[i,j] = dist
        distance_matrix[j,i] = distance_matrix[i,j]

ax_d.imshow(distance_matrix)
ax_d.set_title('Densenet features')
ax_d.grid(True)

plt.show()
