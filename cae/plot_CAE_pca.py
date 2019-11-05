import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets
from torchvision import transforms

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from models import Average_CAE_deep
from models import Average_CAE_deep_PCA


from train import *

# Dataloader parameters
batch_size = 64
validation_split = .2
random_seed= 42
# random_seed= 13

# Transfer Learning
pretrained_weights = True
freeze_pretrained_weights = True
# unfreeze_epoch = num_epochs // 2
unfreeze_epoch = 10


# Paths
# dell
# input_images_path = '../../patches_images/test/'
# save_checkpoint_path = '/home/fede/Documents/mhpc/mhpc-thesis/code/patches_models'
# load_checkpoint_file = '/home/fede/Documents/mhpc/mhpc-thesis/code/patches_models/20191007-152356_Average_CAE-128x8x8.pt'

# Ulysses
# input_images_path = '/scratch/fbarone/test_256/'
# output_images_path = './output/images'
# save_checkpoint_path = '/scratch/fbarone/cae_models'
# load_checkpoint_file = '/scratch/fbarone/cae_models/20191018-210808_Average_CAE_deep_PCA-1024.pt'

# ingres/picasso
input_images_path = '/scratch/fbarone/test_256/'
output_images_path = './output/images'
activations_path = './output/activations'
save_checkpoint_path = './output/cae_models'

# 256x8x8
pretrained_model_256x8x8 = './output/cae_models/20191009-232545_Average_CAE_deep-256x8x8.pt' # general weights
#1024
pretrained_model_1024 = './output/cae_models/20191018-210808_Average_CAE_deep_PCA-1024.pt'
#100
pretrained_model_100 = './output/cae_models/20191022-192609_Average_CAE_deep_PCA-100-256x8x8.pt'

# pca_filepath = 'output/activations/pca-100_coding_Average_CAE_deep-256x8x8.pkl' # load pca weights


print("----------------------------------------------------------------")
print("Dataset and Dataloaders: \n")

# Transformations to be compatible with Densenet-121 from NYU paper.
# Note I am using mean and std as recommended in Pytorch. Maybe calculating the dataset statistics is better.
composed = transforms.Compose([ 
#                                 mm_patch.transforms.ToImage(),
                                transforms.ToTensor(),
                                mm_patch.transforms.Scale()
#                                 mm_patch.transforms.GrayToRGB(),
#                                 transforms.Normalize(mean=[18820.3496], std=[8547.6963])
                            ])

# Dataset
# patches = datasets.ImageFolder('/scratch/fbarone/test_256/', transform = composed, target_transform=None, loader=read_image_png)
patches = datasets.ImageFolder(input_images_path, transform = composed, target_transform=None, loader=read_image_png)


# Creating data indices for training and validation splits:
dataset_size = len(patches)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
print(f'dataset size: {dataset_size}')
print(f'training size: {dataset_size - split}')
print(f'validation size: {split} ({validation_split}%)')
shuffle_dataset = True
if shuffle_dataset :
    print("shuffling image indices: True")
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    print(indices[:10])
else:
    print(indices[:10])

train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
# the sampler is needed to sample from the defined list of indices
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# this train loader is to have random access
# loader = torch.utils.data.DataLoader(patches, batch_size=batch_size, 
#                    torch.utils.data.                         num_workers=10)
train_loader = DataLoader(patches, batch_size=batch_size, sampler=train_sampler, 
                            num_workers=10, pin_memory=True, drop_last=True)
valid_loader = DataLoader(patches, batch_size=batch_size, sampler=valid_sampler, 
                            num_workers=10, pin_memory=True, drop_last=True)
# valid_loader = DataLoader(patches, batch_size=batch_size,
#                             num_workers=10, pin_memory=True, drop_last=True)

print("----------------------------------------------------------------")


# check for nvidia library
assert torch.has_cudnn == True, 'No cudnn library!'

# set device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f'device: {device}')


# initialize the NN model
model_256x8x8 = Average_CAE_deep().to(device)
model_1024 = Average_CAE_deep_PCA(1024).to(device)
model_100 = Average_CAE_deep_PCA(100).to(device)

summary(model_1024, input_size=(1, 256, 256), device = 'cpu')


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


# evaluate on the test dataset
# criterion = nn.MSELoss()
# valid_loss = valid(model, criterion, valid_loader, device, 10)

# Plot results
# obtain one batch of test images
num_patches = 10
rows = 4
cols = 10
dataiter = iter(valid_loader)
images, labels = dataiter.next()
images = images[:num_patches]
len(images)




# get sample outputs
model_256x8x8.eval()
model_1024.eval()
model_100.eval()

output_256x8x8 = model_256x8x8(images.to(device=device))
output_1024 = model_1024(images.to(device=device))
output_100 = model_100(images.to(device=device))

# prep images for display
images = images.cpu().numpy()

# output is resized into a batch of images
output_256x8x8 = output_256x8x8.view(num_patches, 1, 256, 256)
output_1024 = output_1024.view(num_patches, 1, 256, 256)
output_100 = output_100.view(num_patches, 1, 256, 256)
# use detach when it's an output that requires_grad
output_256x8x8 = output_256x8x8.detach().cpu().numpy()
output_1024 = output_1024.detach().cpu().numpy()
output_100 = output_100.detach().cpu().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(25,10))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output_256x8x8, output_1024, output_100], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)


axes[0,0].set_ylabel('Input', fontsize = 12)
axes[1,0].set_ylabel('CAE [CF=4]')
# axes[1,cols - 1].set_ylabel('CAE')
# axes[1,cols - 1].yaxis.set_label_position("right")
axes[2,0].set_ylabel('CAE+PCA-1024 [CF=64]')
# axes[2,cols - 1].set_ylabel('CAE + PCA-1024')
# axes[2,cols - 1].yaxis.set_label_position("right")
axes[3,0].set_ylabel('CAE+PCA-100 [CF=640]')
# axes[3,cols - 1].set_ylabel('CAE + PCA-100')
# axes[3,cols - 1].yaxis.set_label_position("right")

fig.text(0.1, 0.5, '+ Compression Factor -', va='center', rotation='vertical', fontsize = 12)

plt.show()


# # output figures

# fig_path = os.path.join( output_images_path, checkpoint['name'][:-3] + '.png' )
# fig_hist_path = os.path.join( output_images_path ,checkpoint['name'][:-3] + '_hist.png')
# fig.savefig(fig_path, bbox_inches='tight')
# fig_hist.savefig(fig_hist_path, bbox_inches='tight')
