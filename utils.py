import scipy.misc
import scipy.ndimage
from PIL import Image
from skimage import io,transform,feature
import numpy as np
import glob
import os
import h5py
import torch
import pprint
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn



def imread(path, as_gray=True):
    """Read image using skimage module"""
    image = io.imread(path, as_gray=as_gray) # as_gray=True means to flatten the color layers into a single gray-scale layer
    if as_gray:
        image = image[:, :, np.newaxis] # a ndarray with shape MxNx1
    # else return a ndarray with shape MxNx3
    return image 

def imwrite(path, img):
    io.imsave(path, img)

def compute_par(images1, images2, mode='edge'):
    """Compute the pixel agreement ratio of two batch of images
    Parameters
    ----------
    images1: torch.Tensor
        A batch of images from pipeline 1, with shape [batch_size, C, H, W]
    images2: torch.Tensor
        Another batch of images from pipeline 2, with shape [batch_size, C, H, W]
    Returns
    -------
    par: float
        The computed pixel agreement ratio
    Test example
    ------------
    images1 = torch.IntTensor([ [ [ [1,2,1], [1,1,1], [1,2,1] ] ] ])
    images2 = torch.IntTensor([ [ [ [254,254,254], [0,254,0], [254,254,254] ] ] ])
    print('test int input(neg):', compute_par(images1, images2, 'negative')) # 0.555
    images1 = torch.FloatTensor([ [ [ [0.3,1,0.3], [0.3,0.3,0.3], [0.3,1,0.3] ] ] ])
    images2 = torch.FloatTensor([ [ [ [0.7,0.7,0.7], [0,0.7,0], [0.7,0.7,0.7] ] ] ])
    print('test float input(neg):', compute_par(images1, images2, 'negative')) # 0.555
    im = np.zeros((16, 16))
    im[4:-4, 4:-4] = 1
    im += 0.2 * np.random.rand(*im.shape)
    images1 = torch.FloatTensor(im[np.newaxis, np.newaxis, :, :])
    images2 = torch.FloatTensor(feature.canny(im, sigma=3) * im)
    print('test float input(edge):', compute_par(images1, images2, 'edge')) # 1.0
    """    
    # def compute_par_uint(images1, images2):
    #     """Compute the pixel agreement ratio of two image with uint8 dtype"""
    #     diff = images1 - images2
    #     par_uint = diff[diff == 0].numel() / images1.numel()
    #     return par_uint
    def compute_par_float(images1, images2):
        """Compute the pixel agreement ratio of two image with float dtype"""
        diff = (images1 - images2).abs()
        par_uint = diff[diff < 1e-4].numel() / images1.numel() # any threshold less than 1/2/255 is ok
        return par_uint
    if images1.dtype == torch.int32: # torch.int32 is the default type of integer tensor
        images1 = images1.float() / 255
        images2 = images2.float() / 255
    # must be float32 when arrive here
    if mode == 'edge':
        images2_goal = np.zeros_like(images1)
        images1 = images1.numpy()
        for i in range(images1.shape[0]):
            # the dtype of output of feature.canny is bool, so it need multiplication
            images2_goal[i, 0, :, :] = feature.canny(images1[i, 0, :, :], sigma=3) * images1[i, 0, :, :]
        images2_goal = torch.FloatTensor(images2_goal)
    elif mode == 'negative':
        images2_goal = 1 - images1       
    par = compute_par_float(images2_goal, images2)

    return par

# def check_params(model):
#     """
#     Check parameters in models, especially used in weight-share model to verify that
#     the weight is shared correctly.
#     Shared weight have the same summation, while independent one are not the same.
#     """
#     for name, param in model.named_parameters():
#         print(name, param.sum())

def check_params(model1, model2=''):
    """
    Single model version (model2 has passed nothing):
        Check parameters in models, especially used in weight-share model to verify that
        the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    Double model version:
        Check parameters in two weight-share models, to verify that the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    """    
    if model2 == '':
        for name, param in model1.named_parameters():
            print(name, param.sum())
            # print(name, param.size(), param.sum())
        return 

    # simple implementation, hard to make comparation if the model is a bit complex
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    
    # complex implememtation
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()        
    m1_next, m2_next = True, True
    while m1_next or m2_next:
        try:
            name, param = params1.__next__()
            print(name, param.sum())
        except StopIteration:
            # print('stop1')
            m1_next = False
        try:
            name, param = params2.__next__()
            print(name, param.sum())
        except StopIteration:
            # print('stop2')
            m2_next = False            


# def summary(input_size, model):
def check_model(input_size, model):
    """
    Use a random noise input tensor to flow through the entire network, to test
    if the demension of all modules is matching    
    Beacuse if there exist fc layer in the model, the H and W of test input is unique, and 
    the in_channels of fc layer is hard to obtain, so give up the idea to automatically generate
    a test input, but use a external input instead.
    Parameters
    ----------
    input_size: list or tuple
        the size of input tensor, which is [C, H, W]
    model: a subclass of nn.Module
        the model need to be checked
    Returns
    -------
    summary: OrderedDict, optional(disabled now)
        Contain informations of base modules of the model, each module has info about 
        input_shape, output_shape, num_classes, and trainable
    Usage
    -----
    model = Model(args)
    check_model(input_size, model)
    """
    def get_output_size(summary_dict, output):
        if isinstance(output, tuple): # if the model has more than one output, "output" here will be a tuple
            for i in range(len(output)): 
                summary_dict[i] = OrderedDict()
                summary_dict[i] = get_output_size(summary_dict[i],output[i])
        else:
            summary_dict['output_shape'] = list(output.size())
        return summary_dict

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            # print(module.__class__)
            module_idx = len(summary)
            # print(classes_idx)
            allow_base_modules_only = True # it control whether create summary for those middle modules
            if allow_base_modules_only:
                base_classes = ['Linear', 'Conv2d', 'Flatten', 'ReLU', 'PReLU'] # 有待添加，随着网络的变化而变化
                if class_name not in base_classes:
                    return 
            class_idx = classes_idx.get(class_name)
            if class_idx == None:
                class_idx = 0
                classes_idx[class_name] = 1
            else:
                classes_idx[class_name] += 1
            # print(type(input), type(output))
            m_key = '%s-%i (%i)' % (class_name, class_idx+1, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size()) # input is a tuple having an tensor element
            summary[m_key] = get_output_size(summary[m_key], output)
        
            params = 0
            if hasattr(module, 'weight'):
                params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            #if hasattr(module, 'bias'):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
    
            summary[m_key]['num_params'] = params # not take bias into consideration
            pprint.pprint({m_key: summary[m_key]})
            # print(m_key, ":", summary[m_key]) # print info of each module in one line
        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not (module == model): # make sure "module" is a base module, such conv, fc and so on
            hooks.append(module.register_forward_hook(hook)) # hooks is used to record added hook for removing them later
  
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size)) # 1 is batch_size dimension to adapt the model's structure

    # create properties
    summary = OrderedDict()
    classes_idx = {}
    hooks = []
    # register hook
    model.apply(register_hook) # 递归地去给每个网络组件挂挂钩（不只是conv, fc这种底层组件，上面的Sequential组件也会被访问到）
    # make a forward pass
    output = model(x)    
    # output, _ = model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    # pprint.pprint(summary)
    print('output size:', output.size())
    print('Check done.')
    # return summary    