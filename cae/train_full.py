import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader 
import torch.optim as optim
# Where to add a new import
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets
from torchvision import transforms

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler

import models
from utils import ExpoAverageMeter
from utils import read_image_png
from models import Basic_CAE
from models import DenseNet_CAE_gen
from train import train_one_epoch_DCAE
from train import valid_DCAE

sys.path.insert(0,'/u/f/fbarone/Documents/patches/')

import mm_patch.transforms 
from mm_patch.utils import image_from_index

# # Build DenseNet_CAE model
# dense_param = {'num_classes': 4}
# gen_param = {'nc': 1, 'nz': 1024, 'ngf': 256, 'ngpu': 1}
# model = models.DenseNet_CAE(dense_param, gen_param).to(device)

# ## Load subset of pretrained model
# model_path = '/home/fede/Documents/mhpc/mhpc-thesis/code/breast_cancer_classifier/models/sample_patch_model.p'
# # densenet121 dict (subnet inside DenseNet_CAE)
# # densenet121_dict = model.densenet121.densenet.state_dict()

# # pretrained model_dict
# pretrained_model_dict = torch.load(model_path)# ["model"]

# # load pretrained parameters into densenet121
# model.densenet121.densenet.load_state_dict(pretrained_model_dict)

# exit()


## Start Training
# Dataloader parameters
batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Transformations to be compatible with Densenet-121 from NYU paper.
# Note I am using mean and std as recommended in Pytorch. Maybe calculating the dataset statistics is better.
composed = transforms.Compose([ 
#                                 mm_patch.transforms.ToImage(),
                                transforms.ToTensor(),
                                mm_patch.transforms.Scale(),
                                mm_patch.transforms.GrayToRGB(),
                                # Norm does (image - mean) / std
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
                                # true (mean,std) of patches : (0.3373, 0.1767)
                            ])



# Dataset
# patches = datasets.ImageFolder('/scratch/fbarone/patches_256_CRO_23072019', transform = composed, target_transform=None, loader=read_image_png)
patches = datasets.ImageFolder('/scratch/fbarone/test_256', transform = composed, target_transform=None, loader=read_image_png)


# Creating data indices for training and validation splits:
dataset_size = len(patches)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# this train loader is to have random access
# loader = torch.utils.data.DataLoader(patches, batch_size=batch_size, 
#                    torch.utils.data.                         num_workers=10)
train_loader = DataLoader(patches, batch_size=batch_size, sampler=train_sampler, 
                            num_workers=10, pin_memory=True, drop_last=True)
valid_loader = DataLoader(patches, batch_size=batch_size, sampler=valid_sampler, 
                            num_workers=10, pin_memory=True, drop_last=True)

# check for nvidia library
# assert torch.has_cudnn == True, 'No cudnn library!'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the NN model
# Build DenseNet_CAE model
dense_param = {'num_classes': 4}
gen_param = {'nc': 1, 'nz': 1024, 'ngf': 256, 'ngpu': 1}
model = DenseNet_CAE_gen(dense_param, gen_param).to(device)

## Load subset of pretrained model
model_path = '/u/f/fbarone/Documents/breast_cancer_classifier/models/sample_patch_model.p'
# densenet121 dict (subnet inside DenseNet_CAE)
# densenet121_dict = model.densenet121.densenet.state_dict()

# pretrained model_dict
# pretrained_model_dict = torch.load(model_path)# ["model"]

# load pretrained parameters into densenet121
# model.densenet121.densenet.load_state_dict(pretrained_model_dict)

model.densenet121.load_from_path(model_path)

# freeze densenet121 layer
for param in model.densenet121.parameters():
    param.requires_grad = False

# print summary]
# summary(your_model, input_size=(channels, H, W))
summary(model, input_size=(3, 256, 256), device = 'cuda')


# Training parameters
num_epochs = 100
lr = 0.002

# number of checpoints
checkpoint_freq = 1

# loss function
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

# optimizer algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters())

print("----------------------------------------------------------------")

print('\n')
print("----------------------------------------------------------------")
print("Start training...")
print(f'num_epochs: {num_epochs}, \
        initial lr: {lr} \
        batch_size: {batch_size}')
print("================================================================")


# factor = decaying factor
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

loss_hist = []
for it, epoch in enumerate(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    train_loss = train_one_epoch_DCAE(model, optimizer, criterion, train_loader, \
                    device, epoch, print_freq = 2)


    # evaluate on the test dataset
    valid_loss = valid_DCAE(model, criterion, valid_loader, device, epoch)
    
    # keep track of train and valid loss history
    loss_hist.append((train_loss.val, valid_loss.val))

    # update the learning rate
    scheduler.step(train_loss.val)

    print("-----------------")

    # checkpoint
    # if (it + 1) % (num_epochs // checkpoint_freq) == 0:

print("================================================================")

# Plot results
# obtain one batch of test images
num_patches = 10
dataiter = iter(valid_loader)
images, labels = dataiter.next()
images = images[:num_patches]
len(images)
# get sample outputs
output = model(images.to(device=device))
# prep images for display
images = images.cpu().numpy()
images = images[:,0,:,:][:,None,:,:]

# output is resized into a batch of images
print(len(output))
print(output[0].shape)
output = output.view(num_patches, 1, 256, 256)
# use detach when it's an output that requires_grad
output = output.detach().cpu().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# Training history
fig_hist = plt.figure()
ax = fig_hist.add_subplot(1,1,1)
# unzip list of tuples
train_loss, valid_loss = zip(*loss_hist)
ax.plot(range(len(loss_hist)), train_loss, label = 'training loss')
ax.plot(range(len(loss_hist)), valid_loss, label = 'validation loss')
ax.set_xlabel('epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.show()