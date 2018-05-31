import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# A refactoring job would be to make this class simpler 
class UNetBlock( nn.Module ):
    def __init__(self, in_chan, out_chan, inner_in_chan, num_z=0, submodule=None, outermost=False,innermost=False, norm_layer=None, lin_layer=None, dropout=False, upsample='basic', padding_type='zero'):
        super( UNetBlock, self).__init__()
        pad = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            pad = 1
        self.innermost = innermost
        self.outermost = outermost

class UNet( nn.Module ):
    def __init__(self, in_chan, out_chan, num_z, num_down, filters, norm_layer=None, lin_layer=None, dropout=False, gpu_ids=[], upsample='basic'):
        super(UNet,self).__init__()

        block = UNetBlock() #TODO: Continue