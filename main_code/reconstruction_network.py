# from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys,glob
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import functools

from torchvision import models
import segmentation_models_pytorch as smp
# from main_reconstruction_spertral_normal import SpectralNorm
from torch.autograd import Variable
import torch.autograd as autograd
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
        # print(x.shape)
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



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
#################################################################
#                        simple unet                            #
#################################################################
class reconstruction_resunet(nn.Module):
    def __init__(self,in_channels=2,classes=2,multi_output=False):
        super(reconstruction_resunet,self).__init__()
        # se_resnet101
        # resnet101
        # self.model = smp.Unet('se_resnet101',in_channels=in_channels,classes=classes,activation='sigmoid',encoder_weights=None)
        self.model = smp.Unet('se_resnet50',in_channels=in_channels,classes=classes,activation='arctan',encoder_weights=None)
        # self.model = smp.Unet('se_resnet101',in_channels=in_channels,classes=classes,activation='nn.Tanh(),encoder_weights=None)
    def forward(self,x):
        return self.model(x)

class reconstruction_deeplab(nn.Module):
    def __init__(self,in_channels=2,classes=2,multi_output=False):
        super(reconstruction_deeplab,self).__init__()
        # se_resnet101
        # resnet101
        # arctan
        # self.model = smp.Unet('se_resnet101',in_channels=in_channels,classes=classes,activation='sigmoid',encoder_weights=None)
        self.model = smp.DeepLabV3('se_resnet50',in_channels=in_channels,classes=classes,activation='arctan',encoder_weights=None)
        # self.model = smp.Unet('se_resnet101',in_channels=in_channels,classes=classes,activation='nn.Tanh(),encoder_weights=None)
    def forward(self,x):
        return self.model(x)


class reconstruction_discrim(nn.Module):
    def __init__(self,in_channels=1,classes=1,multi_output=False):
        super(reconstruction_discrim,self).__init__()
        self.model = smp.Unet('resnet101',in_channels=in_channels,classes=classes,activation='sigmoid',encoder_weights=None)
        # self.model = smp.Unet('resnet34',in_channels=in_channels,classes=classes,activation={'arctan':torch.nn.Tanh()},encoder_weights=None)
        self.last = nn.Conv2d(1024,1,3,padding=1)

    def forward(self,x):
        result = self.model.encoder.forward(x)
        # print(result[len(result)-2].shape)
        result[len(result)-2] = self.last(result[len(result)-2])
        # print(result[len(result)-2].shape)
        return result[len(result)-2]

class classification_discrim(nn.Module):
    def __init__(self,in_ch=1):
        super(classification_discrim,self).__init__()
        self.first = nn.Conv2d(in_ch,3,3,1,padding=1)
        self.discrim = models.resnet101(pretrained=True)
        self.last = nn.Linear(1000,1)

    def forward(self,x):
        x = self.first(x)
        result = self.discrim(x)
        result = self.last(result)
        return result

#################################################################
#                     efficient_unet++                          #
#################################################################
class reconstruction_efficientunet(nn.Module):
    def __init__(self,in_channels=2,classes=2,multi_output=False):
        super(reconstruction_efficientunet,self).__init__()
        #unet forward
        # feature_ = [40,32,48,136,384]
        
        feature_ = [64,256,512,1024,2048]

        # self.model =smp.Unet('se_resnet101',in_channels=in_channels,classes=classes,encoder_weights=None)
    
        self.model =smp.Unet('se_resnet50',in_channels=in_channels,activation='arctan',classes=classes,encoder_weights=None)
    
        delayers = list(self.model.decoder.children())[1]
        # delayers = self.model.decoder(activation='arctan')
        # print(delayers)
        self.deconv5,self.deconv4,self.deconv3,self.deconv2,self.deconv1 = delayers
        # print(self.deconv1)
        # print(delayers)
        # self.deconv5 = self.deconv5(activation='arctan')
        # print(deconv5)
        self.deconv1 = VGGBlock(32,16,16)
        # self.finals = self.model.segmentation_head
        
        self.up_x2_1 = single_conv(feature_[3]+feature_[2]  ,feature_[2])
        self.up_x1_2 = single_conv(feature_[2]+feature_[1]*2,feature_[1])
        self.up_x0_3 = single_conv(feature_[1]+feature_[0]*3,feature_[0])
        self.topconcat = single_conv(feature_[0]*4,feature_[0])

        self.up_x1_1 = single_conv(feature_[2]+feature_[1]  ,feature_[1])
        self.up_x0_2 = single_conv(feature_[1]+feature_[0]*2,feature_[0])
        self.midconcat = single_conv(feature_[1]*3,feature_[1])

        self.up_x0_1 = single_conv(feature_[1]+feature_[0]  ,feature_[0])
        self.lastconcat = single_conv(feature_[2]*2,feature_[2])
        
        if multi_output == True:
            self.final1 = nn.Conv2d(feature_[0], classes, kernel_size=1)
            self.final2 = nn.Conv2d(feature_[0], classes, kernel_size=1)
            self.final3 = nn.Conv2d(feature_[0], classes, kernel_size=1)
        self.multi_output = multi_output
    
        self.finals = nn.Conv2d(16, classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.softmax = nn.(dim=1)
        self.sigmoid = nn.Tanh()
    def encoderforward(self,x):
        return self.model.encoder.forward(x)
    def forward(self,x):
        _,x0_0,x1_0,x2_0,x3_0,x4_0 = self.encoderforward(x)
        
        #first skip connection
        # print(x0_0.shape,x1_0.shape,x2_0.shape,x3_0.shape,x4_0.shape)
        x0_1 = self.up_x0_1(torch.cat([x0_0,self.upsample(x1_0)],1))

        #second skip connection
        x1_1 = self.up_x1_1(torch.cat([x1_0,self.upsample(x2_0)],1))
        x0_2 = self.up_x0_2(torch.cat([x0_0,x0_1,self.upsample(x1_1)],1))

        #third skip connection
        x2_1 = self.up_x2_1(torch.cat([x2_0,self.upsample(x3_0)],1))
        x1_2 = self.up_x1_2(torch.cat([x1_0,x1_1,self.upsample(x2_1)],1))
        x0_3 = self.up_x0_3(torch.cat([x0_0,x0_1,x0_2,self.upsample(x1_2)],1))

        #forth depth
        x3_1 = torch.cat([x3_0,self.upsample(x4_0)],1)
        x3_1 = self.deconv5(x3_1)

        #third depth
        x2_0 = self.lastconcat(torch.cat([x2_0,x2_1],1))
        x2_2 = torch.cat([x2_0,x3_1],1)
        x2_2 = self.deconv4(x2_2)

        #second depth
        x1_0 = self.midconcat(torch.cat([x1_0,x1_1,x1_2],1))
        x1_3 = torch.cat([x1_0,x2_2],1)
        x1_3 = self.deconv3(x1_3)

        #first depth
        x0_0 = self.topconcat(torch.cat([x0_0,x0_1,x0_2,x0_3],1))
        x0_4 = torch.cat([x0_0,x1_3],1)
        x0_4 = self.deconv2(x0_4)
        
        last = self.deconv1(x0_4)
        # print(last.shape,'111')
        last = self.finals(last)
        
        
        if self.multi_output == True:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            result= self.finals(last)
            return [output1,output2,output3,result]
        return self.sigmoid(last)


# def compute_gradient_penalty(D, real_samples, fake_samples):
#     cuda = True if torch.cuda.is_available() else False
#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).expand_as(real_samples)
#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = Variable(Tensor(real_samples.shape[0],1).fill_(1.0), requires_grad=False)
#     # Get gradient w.r.t. interpolates
#     gradients = autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

class pyramid_unet(nn.Module):
    def __init__(self,in_channels=2,classes=2,multi_output=False):
        super(pyramid_unet,self).__init__()
        self.model = smp.Unet('se_resnet152',in_channels=in_channels,classes=classes,activation='softmax',encoder_weights=None)
    def forward(self,x):
        return self.model(x)


#################################################################
#                using perceptual loss++                        #
#################################################################


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.first = nn.Conv2d(2,3,3,padding=1)
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        
        h = self.first(x)
        
        h = self.to_relu_1_2(h)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # print(h_relu_1_2.shape, h_relu_2_2.shape, h_relu_3_3.shape, h_relu_4_3.shape)
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out