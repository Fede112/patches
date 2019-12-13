import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets
from torchvision import transforms

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler

from models import DenseNet_CAE_PCA


from train import *

import models
from utils import ExpoAverageMeter
from utils import read_image_png
from models import DenseNet_CAE_gen
from train import train_one_epoch_DCAE
from train import valid_DCAE

sys.path.insert(0,'/u/f/fbarone/Documents/patches/')

import mm_patch.transforms 
from mm_patch.utils import image_from_index

# Dataloader parameters
batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42
# random_seed= 13

# Training parameters
num_epochs = 30
lr = 0.0002
# lr = 0.002
# number of checkpoints
checkpoint_freq = 1

# Transfer Learning
pretrained_weights = True
freeze_pretrained_weights = True
# unfreeze_epoch = num_epochs // 2
unfreeze_epoch = 5


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


pretrained_densenet_filepath = '/u/f/fbarone/Documents/breast_cancer_classifier/models/sample_patch_model.p'
# load_checkpoint_file = './output/cae_models/20191009-232545_Average_CAE_deep-256x8x8.pt' # general weights
pretrained_decoder_filepath = './output/cae_models/20191018-210808_Average_CAE_deep_PCA-1024.pt' # load pca weights
pca_filepath = 'output/activations/pca-1024_coding_Average_CAE_deep-256x8x8.pkl' # load pca weights


print("----------------------------------------------------------------")
print("Dataset and Dataloaders: \n")

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
# patches = datasets.ImageFolder('/scratch/fbarone/test_256/', transform = composed, target_transform=None, loader=read_image_png)
patches = datasets.ImageFolder(input_images_path, transform = composed, target_transform=None, loader=read_image_png)


# Creating data indices for training and validation splits:
dataset_size = len(patches)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
print(f'dataset size: {dataset_size}')
print(f'training size: {dataset_size - split}')
print(f'validation size: {split} ({validation_split}%)')
if shuffle_dataset :
    print("shuffling image indices: True")
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

# images = next(iter(train_loader))
# img = images[0][0,0,:,:]
# print(np.amin(np.array(img)))
# print(images[0][0,0,:,:].shape)

print("----------------------------------------------------------------")


# check for nvidia library
assert torch.has_cudnn == True, 'No cudnn library!'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'device set to: {device}')

# Initialize the NN model
# Build DenseNet_CAE model
dense_param = {'num_classes': 4}
model = DenseNet_CAE_PCA(dense_param).to(device)

# pretrained weights
pretrained_model_dict = {}
if pretrained_weights:
    print("----------------------------------------------------------------")
    print("Loading pretrained weights... ", end='')


    ## Load Densenet model
    model.densenet121.load_from_path(pretrained_densenet_filepath)

    ## Load Decoder from Average_CAE_deep_PCA
    ## Load subset of pretrained model
    # sub_model keys
    decoder_dict = model.decoder.state_dict()
    # load full model pretrained dict
    pretrained_decoder_dict = torch.load(pretrained_decoder_filepath)['state_dict']
    pretrained_pca = torch.load(pca_filepath)


    # From pytorch discuss: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16
    # 1. filter out unnecessary keys
    pretrained_decoder_dict = {k: v for k, v in pretrained_decoder_dict.items() if k in decoder_dict}
    pretrained_pca = {k: v for k, v in pretrained_pca.items() if k in decoder_dict}


    # 2. overwrite entries in the existing state dict
    decoder_dict.update(pretrained_decoder_dict) 
    decoder_dict.update(pretrained_pca) 
    # 3. load the new state dict
    # model.decoder.load_state_dict(decoder_dict)


    print("Done!")

    print("----------------------------------------------------------------")



# loss function
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

# optimizer algorithm
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters())

# factor = decaying factor
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# training and validation loss recorder
loss_hist = []



print('\n')
print("----------------------------------------------------------------")
print("Start training...")
print(f'num_epochs: {num_epochs} \
        initial lr: {lr} \
        batch_size: {batch_size}')
print("================================================================")



if freeze_pretrained_weights:
    print("----------------------------------------------------------------")
    print("Initially freezed Densenet parameters:")
    # freeze densenet121 layer
    for name, param in model.densenet121.named_parameters():
        param.requires_grad = False

    # freeze pca linear layer of decoder
    # for name,param in model.decoder.pca_decoder.named_parameters():
    #     print(name)
    #     param.requires_grad = False        
    print("----------------------------------------------------------------")
        
# print summary
# summary(your_model, input_size=(channels, H, W))
summary(model, input_size=(3, 256, 256), device = 'cuda')

# for name, child in model.named_children():
#     for child 
#     if name in pretrained_model_dict.keys():
#         print(name + ' is frozen')
#         for param in child.parameters():
#             param.requires_grad = False
#     else:
#         print(name + ' is un frozen')
#         for param in child.parameters():
#             print(param.name)
#             param.requires_grad = True


for it, epoch in enumerate(range(num_epochs)):

    # Unfreeze pretrained weights
    if epoch == unfreeze_epoch and freeze_pretrained_weights:
        print("Unfreezing the following parameters:")
        # Densenet121
        # for name, param in model.densenet121.named_parameters():
        #     param.requires_grad = True
        #     # model.param.requires_grad = False
        #     print(name)           
    
        # linear layer of CAE-PCA decoder
        # for name,param in model.decoder.pca_decoder.named_parameters():
        #     print(name)
        #     param.requires_grad = True        
    

        # summary(model, input_size=(3, 256, 256), device = 'cuda')

        
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
    if (it + 1) % (num_epochs // checkpoint_freq) == 0:
        # pass
        checkpoint = save_checkpoint(epoch, model, optimizer, scheduler, criterion, loss_hist, save_checkpoint_path)



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
# plt.show()


# output figures

fig_path = os.path.join( output_images_path, checkpoint['name'][:-3] + '.png' )
fig_hist_path = os.path.join( output_images_path ,checkpoint['name'][:-3] + '_hist.png')
fig.savefig(fig_path, bbox_inches='tight')
fig_hist.savefig(fig_hist_path, bbox_inches='tight')
