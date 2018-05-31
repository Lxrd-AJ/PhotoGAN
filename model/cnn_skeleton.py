import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv

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
        self.num_z = num_z
        in_chan += num_z
        downconv += [nn.Conv2d(in_chan, inner_in_chan, 4, 2, pad)]
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_in_chan) if norm_layer is not None else None 
        uprelu = lin_layer()
        upnorm = norm_layer(out_chan) if norm_layer is not None else None 

        if outermost:
            upconv = upsampleLayer( inner_in_chan * 2, out_chan, upsample=upsample,padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up 
        elif innermost:
            upconv = upsampleLayer(inner_in_chan, out_chan, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up 
        else:
            upconv = upsampleLayer( inner_in_chan * 2, out_chan, upsample=upsample,padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x),x],1)
        


class UNet( nn.Module ):
    def __init__(self, in_chan, out_chan, num_z, num_down, filters, norm_layer=None, lin_layer=None, dropout=False, gpu_ids=[], upsample='basic'):
        super(UNet,self).__init__()

        block = UNetBlock( filters * 8, filters*8, filters*8, num_z, submodule=None, innermost=True, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample)
        block = UNetBlock( filters * 8, filters*8, filters*8, num_z, submodule=block, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample, dropout=dropout)

        for i in range(num_down - 6):
            block = UNetBlock( filters * 8, filters*8, filters*8, num_z, submodule=block, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample, dropout=dropout)

        block = UNetBlock( filters*4, filters*4, filters*8, num_z, submodule=block, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample)
        block = UNetBlock( filters*2, filters*2, filters*4, num_z, submodule=block, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample)
        block = UNetBlock( filters, filters, filters*2, num_z, submodule=block, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample)

        block = UNetBlock( in_chan, out_chan, filters, num_z, submodule=block, outermost=True, norm_layer=norm_layer, lin_layer=lin_layer, upsample=upsample)
        self.model = block

    def forward(self, x, z):
        return self.model(x,z)