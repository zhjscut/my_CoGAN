# def test(train_loader_source):
#     for j in range(4):
#         for i, data in enumerate(train_loader_source):
#             inputs, labels = data
#             print(i, inputs.sum())
#         print('')

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('..')
import numpy as np
from skimage import feature
import time

from opts import opts
from utils import compute_par, check_params, check_model
from data.prepare_data import generate_dataloader, my_dataset
from model import Model


# m = nn.Conv2d(1, 1, 5, padding=2)
# n = nn.Conv2d(1, 1, 5, padding=2)
# # print(dir(m))
# print(m.padding)
# print(m.state_dict()["bias"], n.state_dict()["bias"])
# # state_dict = m.state_dict()["bias"]
# n.load_state_dict(m.state_dict())
# # n.state_dict()["weight"] = m.state_dict()["weight"]
# # n.state_dict()["bias"] = m.state_dict()["bias"]
# print(m.state_dict()["bias"], n.state_dict()["bias"])

# # for name, params in n.named_parameters():

# input = torch.randn(16, 1, 28, 28)
# output = m(input)
# print(output.sum())
# output = n(input)
# print(output.sum())
# a = nn.Conv2d(1,2,5,padding=2)
# b = nn.Sequential(a,a)
# print(b)
# for i, module in enumerate(b.modules()):
#     # print(i, dir(module))
#     # print(i, module.state_dict())
#     state_dict = module.state_dict()
#     print(state_dict)
#     for name in state_dict:
#         # print(state_dict[name])
#         state_dict[name] /= 2

#         # print(data, a)
#         # print(data[0], type(data[1]))
#     # for i, modu in enumerate(module.modules()):
#     #     print(i, modu, dir(modu))
#     break
    
# 不行，这是参数，不是参数的梯度！
# for name in state_dict:
#     state_dict[name] /= 2    

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1) 

class Unflatten(nn.Module):
    # def __init__(self, C=1, H=28, W=28):
    #     self.C, self.H, self.W = C, H, W
    def forward(self, x):
        N = x.shape[0]
        # return x.view(N, C, H, W)         
        return x.view(N, 1, 28, 28)

class Flatten1(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1) 

a = Flatten()
print(a._modules)
print(a.modules())

for i in a._modules:
    print(i)
# b = Flatten1()
b = Unflatten()
for i in b._modules:
    print(i)
input = torch.randn(16, 1, 28, 28)

b(input)