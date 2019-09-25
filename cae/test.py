import torch

from torchsummary import summary

import models


dense_param = {'num_classes': 4}
gen_param = {'nc': 1, 'nz': 1024, 'ngf': 256, 'ngpu': 1}
model = models.DenseNet_CAE(dense_param, gen_param)

# gen_param = {'nc': 1, 'nz': 100, 'ngf': 64, 'ngpu': 1}
# model =models.Generator256(**gen_param)

summary(model, input_size=(3, 256, 256), device = 'cpu')

print(list(model.named_parameters()))

# model = models.Upsample(scale_factor=2, mode = 'nearest')


# prev = torch.randint(0,10,(4,2,2)).float()
# prev = prev[:,None,:,:]

# out = model(prev)

# print(prev)
# print(prev.shape)
# print(out)
# print(out.shape)