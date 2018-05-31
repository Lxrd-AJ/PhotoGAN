import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .cnn_skeleton import UNet

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def initialise_weights(net, init_type, gain=0.02):
    def init_function(module): #module is a pytorch module
        classname = module.__class__.__name__
        # if hasattr(module,'weight') and 
        # TODO: Continue
        print(classname)
    print("Initialising network with " + init_type)
    net.apply(init_function)

def initialise_network(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        net.to( gpu_ids[0] )
        net = torch.nn.DataParallel(net, gpu_ids)
    initialise_weights(net, init_type)
    return net

"""
- UNet_128 is the only supported generated network
- Add z input is only supported for all parts of the network
"""
def define_Gen(input_chan, output_chan, num_z, num_gen_filters, model_net_gen='unet_128', norm='batch', actv ='relu', dropout=False, init_type='xavier', where_add_z='all', upsample='bilinear'): 
    norm_layer =  get_norm_layer(layer_type=norm)
    non_lin_actv = get_non_linearity(layer_type=actv)

    assert num_z > 0 

    net_Gen = #TODO: Continue

    return initialise_network() #TODO
