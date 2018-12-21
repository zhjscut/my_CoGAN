import torch
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict
# try:
#     from resnet_DANN import resnet
# except:
#     from models.resnet_DANN import resnet

class gradient_reversal_layer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None


def Model(args):
    G = Share_convs_G()
    G1 = Generator(G)
    G2 = Generator(G)
    D = Share_convs_D()
    D1 = Discriminator(D)
    D2 = Discriminator(D)
    model_source = CoGAN(args, G1, D1)
    model_target = CoGAN(args, G2, D2) 
    global flag, hooks
    flag = True
    hooks = []
    from utils import check_params, check_model
    # check_params(model_source)
    # check_model([1, 28, 28], model_source)
    model_source.apply(register_hook)
    # D.apply(register_hook)    

    return model_source, model_target


def register_hook(module):
    """Register hooks for the model with some requirements"""
    def show_grad_info(grad_input, grad_output):
        """Show grad_input and grad_output information of a module"""
        for i, grad_out in enumerate(grad_output):
            print('grad_output[%d]:' % (i), grad_out.size(), grad_out.sum())                   
        # if inputs.requires_grad is False, torch will not generate gradient for 
        # the input images, so grad_input[0] of the first layer will be None
        for i, grad_in in enumerate(grad_input):
            try:
                print('grad_input[%d]:' % (i), grad_in.size(), grad_in.sum())        
            except AttributeError:
                print('grad_input[%d]: None' % (i))        

    def hook(module, grad_input, grad_output):
        # print('triggered hook:', str(module.__class__))
        if 'Conv2d' in str(module.__class__):
            # show_grad_info(grad_input, grad_output)

            # print('original gradient:', grad_input[1].sum(), grad_input[2].sum())
            for i in range(grad_input[1].size(0)):
                grad_input[1][i] = grad_input[1][i].mul(0.5)
            grad_input[2][:] = grad_input[2][:].mul(0.5)           
            # print('halved gradient:', grad_input[1].sum(), grad_input[2].sum())

        elif 'Linear' in str(module.__class__):
            # print('original gradient:', grad_input[0].sum(), grad_input[2].sum())
            for i in range(grad_input[0].size(0)): # bias
                grad_input[0][i] = grad_input[0][i].mul(0.5)
            for i in range(grad_input[2].size(0)): # weight
                grad_input[2][i] = grad_input[2][i].mul(0.5)                
            # print('halved gradient:', grad_input[0].sum(), grad_input[2].sum())     

        elif 'PReLU' in str(module.__class__):
            # print('original gradient:', grad_input[1].sum())
            grad_input[1][0] = grad_input[1][0].mul(0.5)
            # print('halved gradient:', grad_input[1].sum())
            
        else:
            show_grad_info(grad_input, grad_output)

        # print('')      

    def register_hook_(submodel): # 原本是直接apply的，但是这样找不到submodel，会把submodel也给hook一遍导致出错，最后只好多包了一层
        def register_hook_base_module(module):
            """Register hooks for all submodules in a Sequential module, without any condition"""
            # global flag, hooks
            # print(module)
            if not isinstance(module, nn.Sequential) \
                and not isinstance(module, nn.ModuleList) \
                and not (module == submodel): # make sure "module" is a base module, such conv, fc and so on
                # print('hook:', str(module.__class__))
                hooks.append(module.register_backward_hook(hook)) # hooks is used to record added hook for removing them later        
        # print(submodel)
        submodel.apply(register_hook_base_module)
        # print('hook done.')


    flag_filter = 1 # 1 to selectively add hook to some modules according to "module_names", and 0 to add hook to all modules
    if flag_filter: # selectively hook
        module_names = ['Share_convs_G', 'Share_convs_D']
        for i in range(len(module_names)):
            module_name = module_names[i]
            if module_name in str(module.__class__):
                print('hook modules in:', str(module.__class__))
                register_hook_(module) 
                break
            if i == len(module_names)- 1: # can not find corrspondent name
                # print('pass:', str(module.__class__))
                pass   
    else: # full hook
        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not (module == model): # make sure "module" is a base module, such conv, fc and so on
            print('hook:', str(module.__class__))
            hooks.append(module.register_backward_hook(hook)) # hooks is used to record added hook for removing them later        
        else:
            print('pass:', str(module.__class__))         


class CoGAN(nn.Module):

    def __init__(self, args, G, D, **kwargs):
        super(CoGAN, self).__init__()
        self.G = G
        self.D = D
        # G.register_forward_hook(hook)

    def forward(self, x, Gz=None):
        """x is a noise input, img is the expected image"""
        if Gz == None: # use both the Generator and Discriminator
            print('x.size:', x.size())
            Gz = self.G(x) # generated image
            print('Gz.size:', Gz.size())
            # Gz = self.flat(Gz)
            out = self.D(Gz)
            print('out.size:', out.size())
        elif type(Gz) == 'torch.FloatTensor': # use the Discriminator only
            out = self.D(Gz)
        else:
            raise TypeError('Gz must be a tensor')
        return out, Gz


class Generator(nn.Module):
    def __init__(self, shared_conv):
        super(Generator, self).__init__()
        self.shared_conv = shared_conv
        self.conv5 = nn.Conv2d(64,1,5, padding=2)
        # self.model = nn.Sequential(OrderedDict([
        #         ('shared_conv', shared_conv),
        #         ('conv5', nn.Conv2d(64,1,5, padding=2))
        #     ])
        # )

    def forward(self, x):
        # out = self.model(x)
        out = self.shared_conv(x)
        out = self.conv5(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, shared_conv):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1,20,5, padding=2)
        self.shared_conv = shared_conv

    def forward(self, x):
        out = self.conv1(x)
        out = self.shared_conv(out)

        return out


class Share_convs_D(nn.Module):
    def __init__(self):
        super(Share_convs_D, self).__init__()
        self.model = nn.Sequential(OrderedDict([
                ('conv2', nn.Conv2d(20,64,5, padding=2)),
                ('prelu2', nn.PReLU()),
                ('conv3', nn.Conv2d(64,128,5, padding=2)),
                ('prelu3', nn.PReLU()),
                ('conv4', nn.Conv2d(128,64,5, padding=2)),
                ('prelu4', nn.PReLU()),
                ('conv5', nn.Conv2d(64,1,5, padding=2)),
                ('prelu5', nn.PReLU()),
                ('flatten1', Flatten()),
                ('fc1', nn.Linear(784,2))
                ])) 

    def forward(self, x):
        out = self.model(x)

        return out


class Share_convs_G(nn.Module):
    def __init__(self):
        super(Share_convs_G, self).__init__()
        self.model = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,20,5, padding=2)),
                ('prelu1', nn.PReLU()),
                ('conv2', nn.Conv2d(20,64,5, padding=2)),
                ('prelu2', nn.PReLU()),
                ('conv3', nn.Conv2d(64,128,5, padding=2)),
                ('prelu3', nn.PReLU()),
                ('conv4', nn.Conv2d(128,64,5, padding=2)),
                ('prelu4', nn.PReLU())
                ]))  
        # self.model = nn.Sequential(
        #         nn.Conv2d(1,20,5),
        #         nn.PReLU(),
        #         nn.Conv2d(20,64,5),
        #         nn.PReLU()
        #         )                      
        # self.conv1 = nn.Conv2d(1,20,5)
        # self.prelu = nn.PReLU()
        # self.conv2 = nn.Conv2d(20,64,5)    
        # self.resnet_conv = resnet_conv
        # self.fc = nn.Linear(convout_dimension, num_class)

    def forward(self, x):
        out = self.model(x)

        # out = self.conv1(x)
        # out = self.prelu(out)
        # out = self.conv2(out)
        # out = self.prelu(out)


        return out


class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1) 


class Unflatten(nn.Module): # just used to test
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, 1, 28, 28)        