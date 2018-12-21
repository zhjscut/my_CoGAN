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
# from test2 import test
args = opts()

torch.manual_seed(777) 
train_loader_source, train_loader_target = generate_dataloader(args)
# print(len(train_loader_source), len(train_loader_target))

model_source, model_target = Model(args)
# check_params(model_source, model_target)
# check_model([1,28,28],model_source)
# torch.manual_seed(777)       
# test(train_loader_source)
# torch.manual_seed(777)       
# test(train_loader_source)
if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

fake_labels = torch.LongTensor(args.batch_size).fill_(0)
real_labels = torch.LongTensor(args.batch_size).fill_(1)
if torch.cuda.is_available():
    fake_labels = fake_labels.cuda()
    real_labels = real_labels.cuda()
fake_labels = Variable(fake_labels)  
real_labels = Variable(real_labels)         
optimizer_source = torch.optim.Adam(model_source.parameters(), lr=args.lr,
                            betas=(args.momentum_1st, args.momentum_2nd),
                            weight_decay=args.weight_decay)
end = time.time()
for i, data in enumerate(train_loader_source):
    print(i)
    inputs, labels = data

    # inputs.requires_grad = True
    # print(inputs.requires_grad)
    # inputs_var, labels_var = Variable(inputs), Variable(labels)
    # inputs_var.requires_grad = True
    # print(inputs_var.requires_grad)

    # print(inputs.size())
    outputs_source, images_source = model_source(inputs)
    outputs_target, images_target = model_target(inputs)
    print("size of outputs:", outputs_source.size())

    loss_G_source = criterion(outputs_source, fake_labels) # 0 means from source domain and 1 means from target domain 
    loss_G_target = criterion(outputs_target, fake_labels)
    print('loss:', loss_G_source, loss_G_target)
    print('backward')
    loss_G_source.backward()
    # print(dir(model_source))
    for name, params in model_source.named_parameters():
        # print(params.grad)
        print(name, params.grad.size(), params.grad.sum())
        # print(name, type(params.grad))
        # pass
    loss_G_target.backward()
    print('source:')
    for name, params in model_source.named_parameters():
        print(name, params.grad.size(), params.grad.sum())
    print('target:')
    for name, params in model_target.named_parameters():
        print(name, params.grad.size(), params.grad.sum())
    # print('grad:', model_source.parameters().grad)
    optimizer_source.step()
    # print(outputs_m1.size(), outputs_m2.size())
    print('batch time:', time.time() - end)
    end = time.time()
    # break # 在迭代的时候突然break有很大概率会触发ConnectionResetError或EOFError



# test(train_loader_source)

# for i, data in enumerate(train_loader_source):
#     print(i)
#     inputs, labels = data
#     outputs, images = model_source(inputs)
#     # print(i, inputs.sum())
# print('')    