import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad

import torchvision.models.densenet as densenet


# DenseNet_CAE
class DenseNet_CAE(nn.Module):
    def __init__(self, dense_param, gen_param):
        super().__init__()

        if isinstance(gen_param, (dict)):
            self.gen_param = gen_param
        else:
            raise TypeError('gen_param must be a dictionary \
                            matching Generator constructor')
        if isinstance(dense_param, (dict)):
            self.dense_param = dense_param
        else:
            raise TypeError('dense_param must be a dictionary \
                            matching ModifiedDenseNet121 constructor')

        self.densenet121 = ModifiedDenseNet121(**dense_param)
        self.generator = Generator256(**gen_param)

    def forward(self, x):
        x = self.densenet121(x)
        x = x[:,:, None, None]
        x = self.generator(x)
        return x




# define the NN architecture
class Basic_CAE(nn.Module):
    def __init__(self):
        super(Basic_CAE, self).__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.Conv2d(128, 128, 3, padding = 1)
        self.t_conv2 = nn.Conv2d(128, 64, 3, padding = 1)
        self.t_conv3 = nn.Conv2d(64, 1, 3, padding = 1)
        self.upsampling = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
#         self.t_conv3 = nn.ConvTranspose2d(4, 2, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))


       
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv3(x))
                
        return x


class Average_CAE(nn.Module):
    def __init__(self):
        super(Average_CAE, self).__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv1_b = nn.Conv2d(32, 32, 3, padding = 1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv2_b = nn.Conv2d(64, 64, 3, padding = 1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv3_b = nn.Conv2d(128, 128, 3, padding = 1)

        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        # self.conv3_b = nn.Conv2d(256, 256, 3, padding = 1)

        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        
        self.t_conv0 = nn.Conv2d(256, 128, 3, padding = 1)

        self.t_conv1 = nn.Conv2d(128, 128, 3, padding = 1)
        self.t_conv1_b = nn.Conv2d(128, 64, 3, padding = 1)
        
        self.t_conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.t_conv2_b = nn.Conv2d(64, 32, 3, padding = 1)
        
        self.t_conv3 = nn.Conv2d(32, 32, 3, padding = 1)
        self.t_conv3_b = nn.Conv2d(32, 1, 3, padding = 1)
        



# self.upsampling = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
        self.upsampling = Upsample(scale_factor=2, mode='nearest')
#         self.t_conv3 = nn.ConvTranspose2d(4, 2, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv1_b(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_b(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv3_b(x))
        # x = self.pool(x)
        # x = F.relu(self.conv4(x))


       
        ## decode ##
        # x = F.relu(self.t_conv0(x))
        # x = self.upsampling(x)
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv1_b(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2_b(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv3(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv3_b(x))
                
        return x


class Average_CAE_deep(nn.Module):
    def __init__(self):
        super().__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)

        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        
        self.conv6 = nn.Conv2d(256, 512, 3, padding = 1)

        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        
        self.t_conv7 = nn.Conv2d(512, 512, 3, padding = 1)

        self.t_conv6 = nn.Conv2d(512, 256, 3, padding = 1)

        self.t_conv0 = nn.Conv2d(256, 256, 3, padding = 1)

        self.t_conv1 = nn.Conv2d(256, 128, 3, padding = 1)
        
        self.t_conv2 = nn.Conv2d(128, 64, 3, padding = 1)
        
        self.t_conv3 = nn.Conv2d(64, 32, 3, padding = 1)

        self.t_conv4 = nn.Conv2d(32, 16, 3, padding = 1)

        self.t_conv5 = nn.Conv2d(16, 1, 3, padding = 1)
        



# self.upsampling = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
        self.upsampling = Upsample(scale_factor=2, mode='nearest')
#         self.t_conv3 = nn.ConvTranspose2d(4, 2, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)

       
        ## decode ##
        x = F.relu(self.t_conv7(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv6(x))
        x = self.upsampling(x)
        # x = F.relu(self.t_conv0(x))
        # x = self.upsampling(x)
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv3(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv4(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv5(x))
                
        return x


# Auxilary for Generator256
# self.iter_example = Interpolate(size=(2, 2), mode='bilinear')
class Upsample(nn.Module):
    """
    Input x should be: (n_batch, n_channels, xsize, ysize)
    Upsample acts upon the shape of each channel, this is, (xsize, ysize)
    Ex: [[1,2],[3,4]] => [[1,1,2,2],[3,3,4,4]]
    """
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        # x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Average_CAE_bn(nn.Module):
    def __init__(self):
        super(Average_CAE_bn, self).__init__()

        self.encoder = nn.Sequential(

            # output: 32x128x128
            nn.Conv2d(1, 32, 3, padding = 1),
            # nn.BatchNorm2d(32, momentum = 0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 32x64x64
            nn.Conv2d(32, 32, 3, padding = 1),
            # nn.BatchNorm2d(32, momentum = 0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 64x32x32
            nn.Conv2d(32, 64, 3, padding = 1),
            # nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 64x16x16
            nn.Conv2d(64, 64, 3, padding = 1),
            # nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 128x8x8
            nn.Conv2d(64, 128, 3, padding = 1),
            # nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 128x8x8
            nn.Conv2d(128, 128, 3, padding = 1),
            # nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2),
            )


        self.decoder = nn.Sequential(

            # Padding compatible with Keras Upsample
            # torch.nn.ZeroPad2d((1, 2, 1, 2)),
            
            # nn.ConvTranspose2d(128, 128, 4), # ~ 8kk parameters
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            # Upsample(scale_factor = 2, mode='nearest'),

            # output: 128x16x16
            nn.Conv2d(128, 128, 3, padding = 1),
            # nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

            
            # 64x32x32
            nn.Conv2d(128, 64, 3, padding = 1),
            # nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),
        
            # 64x64x64
            nn.Conv2d(64, 64, 3, padding = 1),
            # nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

            # 32x128x128
            nn.Conv2d(64, 32, 3, padding = 1),
            # nn.BatchNorm2d(32, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

        
            # 32x256x256
            nn.Conv2d(32, 32, 3, padding = 1),
            # nn.BatchNorm2d(32, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

        
            # 1x256x256
            nn.Conv2d(32, 1, 3, padding = 1),
            # nn.BatchNorm2d(1, momentum = 0.1),

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        # x = 3*x # for nn.Tanh()
        return x




class Generator256(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator256, self).__init__()
        self.main = nn.Sequential(

            # Padding compatible with Keras Upsample
            # torch.nn.ZeroPad2d((1, 2, 1, 2)),
            
            #1024x1x1
            nn.ConvTranspose2d(1024, 512, 4), # ~ 8kk parameters
            # nn.BatchNorm2d(512),
            nn.ReLU(True),

            #256x4x4
            nn.Conv2d(512, 256, 3, padding = 1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

            
            #256x8x8
            nn.Conv2d(256, 128, 3, padding = 1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),
        
            #128x16x16
            nn.Conv2d(128, 64, 3, padding = 1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

            #64x32x32
            nn.Conv2d(64, 32, 3, padding = 1),
            nn.BatchNorm2d(32, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

        
            #32x64x64
            nn.Conv2d(32, 16, 3, padding = 1),
            nn.BatchNorm2d(16, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

        
            #16x128x128
            nn.Conv2d(16, 8, 3, padding = 1),
            nn.BatchNorm2d(8, momentum = 0.1),
            nn.ReLU(True),
            Upsample(scale_factor = 2, mode='nearest'),

            #8x256x256
            nn.Conv2d(8, 1, 3, padding = 1),
            # nn.Tanh()
            # sigmoid activation to output

        )

    def forward(self, x):
        x = self.main(x)
        x = torch.sigmoid(x)
        # x = 3*x # for nn.Tanh()
        return x


class Generator256_bis(nn.Module):
    def __init__(self, input_size=200, alpha=0.2):
        super(Generator, self).__init__()       
        kernel_size = 4
        padding = 1
        stride = 2
        
        # self.input = nn.Linear(input_size, 4 * 4 * 1024)
        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(1024, 512, kernel_size, stride, padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(512, 512, kernel_size, stride, padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(512, 512, kernel_size, stride, padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(512, 256, kernel_size, stride, padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(256, 128, kernel_size, stride, padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(128, 1, kernel_size, stride, padding),
            nn.Tanh()
        )
  
    def forward(self, x):
        x = self.net(x)
        return x




# Generator Code
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.nz = nz # size of latent space
        self.ngpu = ngf # size of feature map in last layer
        self.ngf = nc # number of output channels
        self.ngpu = ngpu # number of gpu available
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )


    def forward(self, input):
        return self.main(input)


# ==============================================================================
# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# You can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# ==============================================================================
"""
Defines the heatmap generation model used in run_producer.py
"""

class ModifiedDenseNet121(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.densenet = densenet.densenet121(*args, **kwargs)
        self._is_modified = False

    def _modify_densenet(self):
        """
        Replace Conv2d and MaxPool2d to resolve the differences in padding 
        between TensorFlow and PyTorch
        """
        assert not self._is_modified
        for full_name, nn_module in self.densenet.named_modules():
            if isinstance(nn_module, (nn.Conv2d, nn.MaxPool2d)):
                module_name_parts = full_name.split(".")
                parent = self._get_module(self.densenet, module_name_parts[:-1])
                actual_module_name = module_name_parts[-1]
                assert "conv" in module_name_parts[-1] or "pool" in module_name_parts[-1]
                setattr(parent, actual_module_name, TFSamePadWrapper(nn_module))
        self._is_modified = True

    def load_from_path(self, model_path):
        self.densenet.load_state_dict(torch.load(model_path))
        self._modify_densenet()

    def forward(self, x):
        if not self._is_modified:
            self._modify_densenet()
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.densenet.classifier(out)
        return out

    @classmethod
    def _get_module(cls, model, module_name_parts):
        obj = model
        for module_name_part in module_name_parts:
            obj = getattr(obj, module_name_part)
        return obj


class TFSamePadWrapper(nn.Module):
    """
    Outputs a new convolutional or pooling layer which uses TensorFlow-style "SAME" padding
    """
    def __init__(self, sub_module):
        super(TFSamePadWrapper, self).__init__()
        self.sub_module = copy.deepcopy(sub_module)
        self.sub_module.padding = 0
        if isinstance(self.sub_module.kernel_size, int):
            self.kernel_size = (self.sub_module.kernel_size, self.sub_module.kernel_size)
            self.stride = (self.sub_module.stride, self.sub_module.stride)
        else:
            self.kernel_size = self.sub_module.kernel_size
            self.stride = self.sub_module.stride

    def forward(self, x):
        return self.sub_module(self.apply_pad(x))

    def apply_pad(self, x):
        pad_height = self.calculate_padding(x.shape[2], self.kernel_size[0], self.stride[0])
        pad_width = self.calculate_padding(x.shape[3], self.kernel_size[1], self.stride[1])

        pad_top, pad_left = pad_height // 2, pad_width // 2
        pad_bottom, pad_right = pad_height - pad_top, pad_width - pad_left
        return pad(x, [pad_top, pad_bottom, pad_left, pad_right])

    @classmethod
    def calculate_padding(cls, in_dim, kernel_dim, stride_dim):
        if in_dim % stride_dim == 0:
            return max(0, kernel_dim - stride_dim)
        return max(0, kernel_dim - (in_dim % stride_dim))