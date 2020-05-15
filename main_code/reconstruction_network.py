# from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys,glob
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import functools
from main_reconstruction_spertral_normal import SpectralNorm

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        TV = []
        # print(x.shape)
        # x = x.permute(0,4,1,2,3)
        for i in range(5):
                # print(x.shape)
                x_a = x[:,i,:,:]
                # x = x.permute(0,3,1,2)
                batch_size = x_a.size()[0]
                h_x = x_a.size()[1]
                # print(h_x)
                w_x = x_a.size()[2]
                count_h = self._tensor_size(x_a[:,1:,:])
                count_w = self._tensor_size(x_a[:,:,1:])
                # print(x_a[:,:,1:,:].shape)
                # print(x_a[:,:,:h_x-1,:].shape)
                h_tv = torch.pow((x_a[:,1:,:]-x_a[:,:h_x-1,:]),2).sum()
                w_tv = torch.pow((x_a[:,:,1:]-x_a[:,:,:w_x-1]),2).sum()
                TV.append(self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size)
        TV = torch.stack(TV)
        return  torch.mean(TV)

    def _tensor_size(self,t):
        return t.size()[0]*t.size()[1]*t.size()[2]


class WGANLoss(nn.Module):
    def __init__(self, loss_type):
        super(WGANLoss, self).__init__()

        if loss_type == 'gen':
            self.forward = self.loss_gen
        else:
            self.forward = self.loss_disc

    def loss_disc(self, outs_disc_fake, out_disc_real):
        return outs_disc_fake.mean() - out_disc_real.mean()

    def loss_gen(self, outs_disc_fake):
        return -1 * outs_disc_fake.mean()

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
###################################################################################
                                    ##RRDB(W/O BATCHNORM)##
###################################################################################
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class reconstruction_resnet101unet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        resnet = models.resnet.resnet34(pretrained=False)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        x = self.first(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Segmentataion_resnet101unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x


        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Segmentataion_resnet101unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class reconstruction_discriminaotr(nn.Module):

    def __init__(self, n_classes=2):
        super().__init__()
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        resnet = models.resnet.resnet34(pretrained=False)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        
        
    def forward(self, x):
        x = self.first(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Segmentataion_resnet101unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
        return x