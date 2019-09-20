# based on https://github.com/foamliu/Autoencoder/blob/master/train.py
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader 

from torchvision import datasets
from torchvision import transforms

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler


from utils import ExpoAverageMeter
from utils import read_image_png
from models import Basic_CAE

sys.path.insert(0,'/u/f/fbarone/Documents/patches/')

import mm_patch.transforms 
from mm_patch.utils import image_from_index

def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, print_freq = 2):
    # Ensure dropout layers are in train mode
    model.train()

    losses = ExpoAverageMeter(alpha = 0.5)  # loss (per word decoded)
    # batch_time = ExpoAverageMeter()  # forward prop. + back prop. time

    # start = time.time()


    # batch forward pass
    for i_batch, data in enumerate(train_loader):
        # _ should be the target, which for cae its the image itself        
        images, _ = data[0].to(device), data[1]
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # keep track of metrics
        losses.update(loss.item()) # update running training loss
        # batch_time.update(time.time() - start) # update time per batch

            
        # Print status
        if (i_batch + 1) % (len(train_loader) // print_freq) == 0:
            print(f'Epoch: [{epoch}] [{i_batch + 1}/ {len(train_loader)}] \t  \
                    Train Loss: {losses.val:.4f} ({losses.avg:.4f})')


    # loss of last mini_batch
    return losses

def valid(model, criterion, val_loader, device, epoch):
    # Ensure dropout layers are in validation mode (no dropout or batchnorm)
    model.eval()

    losses = ExpoAverageMeter(alpha = 0.5)

    with torch.no_grad():
        for data in val_loader:
            images, _ = data[0].to(device), data[1]
            outputs = model(images)
            loss = criterion(outputs, images)

            # keep track of metrics
            losses.update(loss.item()) # update running training loss

        # Print status
        print(f'Epoch: [{epoch}] \t\t Valid Loss: {losses.val:.4f} ({losses.avg:.4f})')


    return losses



def main():

    # Dataloader parameters
    batch_size = 80
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

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
    patches = datasets.ImageFolder('/scratch/fbarone/patches_256_CRO_23072019', transform = composed, target_transform=None, loader=read_image_png)


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

    # initialize the NN model
    model = Basic_CAE().to(device)

    # print summary]
    # summary(your_model, input_size=(channels, H, W))
    summary(model, input_size=(1, 256, 256), device = 'cuda')


    # Training parameters
    num_epochs = 3
    lr = 0.0005
    # number of checpoints
    checkpoint_freq = 1

    # loss function
    criterion = nn.MSELoss()

    # optimizer algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters())


    print("Start training...")
    loss_hist = []
    for it, epoch in enumerate(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, \
                        device, epoch, print_freq = 2)

        # update the learning rate
        # lr_scheduler.step()

        # evaluate on the test dataset
        valid_loss = valid(model, criterion, valid_loader, device, epoch)
        
        # keep track of train and valid loss history
        loss_hist.append((train_loss.val, valid_loss.val))

        # checkpoint
        # if (it + 1) % (num_epochs // checkpoint_freq) == 0:

    print("End training")

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
    train_loss, valid_loss = zip(*loss_hist)
    ax.plot(range(len(loss_hist)), train_loss, label = 'training loss')
    ax.plot(range(len(loss_hist)), valid_loss, label = 'validation loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()