#!/usr/bin/env python
# coding: utf-8
import os,sys,glob
from glob import glob
import cv2
from PIL import Image, ImageSequence
from skimage import io
import numpy as np


from numpy.lib.stride_tricks import as_strided
#import imgaug as ia
#import imgaug.augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from skimage.color import rgb2gray
from torchvision import datasets, transforms
import torch.nn.functional as F

import scipy.ndimage
from skimage.transform import rescale, resize, downscale_local_mean
import torch
import random
import math
from torch.autograd import Variable
# torch.complex = ComplexTensor
batch_size=16
num_image = 5
vali_batch_size=4
i = 0
DIM = 256
# def data_load(imageDir,labelDir,maskDir,batch_size)
np.random.seed(2019)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))


#################################################################
#                      prepcoessing                             #
#################################################################
def zeroone(x,name='Tzeroone'):
    return (x/2.0 + 0.5)

def cvt2tanh(x, name='ToRangeTanh'):
	#with tf.variable_scope(name):
    return (x / 255.0 - 0.5) * 2.0
#############################normalization###################################
def cvt2imag(x, name='ToRangeImag'):
    #with tf.variable_scope(name):
    return (x / 2.0 + 0.5) * 255.0
def npt2imag(x, name='ToRangeImag'):
    # #with tf.variable_scope(name):
    return (x / 2.0 + 0.5) * 255.0
###############################################################################
def cvt2sigm(x, name='ToRangeSigm'):
    #with tf.variable_scope(name):
    return (x / 1.0 + 1.0) / 2.0
#########################change kspcae#########################################
def fft(x):
    return np.fft.fft2(x,norm="ortho")
def ifft(x):
    return np.fft.ifft2(x,norm="ortho")
#########################standardizaion#########################################
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

#################################################################
#                           kspace                              #
#################################################################
def apply_mask(image,mask,name='apply_mask'):
    
    image = image.permute(0,2,3,1)
    mask = mask.permute(0,2,3,1)
    image = torch.fft(image,3).float()
    mask = mask.float()
    masked_image = torch.mul(mask[...,0:1],image)
    recon_img = torch.ifft(masked_image,3)
        
    return masked_image.permute(0,3,1,2),recon_img.permute(0,3,1,2)

def update(recon,image,mask,name='update'):
    

    k_recon,_=apply_mask(recon,torch.ones_like(mask))
    k_image,_=apply_mask(image,torch.ones_like(mask))

    mask = mask.permute(0,2,3,1)
    m_real = mask[...,0:1]
    m_imag = mask[...,0:1]
    m_mask = torch.cat([m_real,m_imag],dim=3)

    # k_return = torch.mul(k_recon,k_image)
    k_image= k_image.float().permute(0,2,3,1)
    k_recon= k_recon.float().permute(0,2,3,1)

    m_mas = m_mask.to(torch.bool)
    # print(m_mas.shape,k_image.shape,k_recon.shape)
    k_return = torch.where(m_mas,k_image,k_recon)

#     k_return = torch.mul(m_mask[...,0:1],k_recon)
    updated = torch.ifft(k_return,3)
    updated = updated.double()
    return updated.float().permute(0,3,1,2)

#################################################################
#                     cross_validation                          #
#################################################################

def divide_kfold(imageDir,labelDir,k=9,name='test'):
    images = np.array(natsorted(glob(imageDir+'*')))
    labels = np.array(natsorted(glob(labelDir+'*')))

    kfold = KFold(n_splits=k)

    train = dict()
    label  = dict()
    i = 0
    for train_index, test_index in kfold.split(images):
        print(f"train_index{train_index} \t test_index:{test_index}")
        img_train,img_test = images[train_index], images[test_index]
        lab_train,lab_test = labels[train_index], labels[test_index]
        i+=1
        train.update([('train'+str(i),img_train),(name+str(i),img_test)])
        label.update([('train'+str(i),lab_train),(name+str(i),lab_test)])
    return train,label


#################################################################
#                    gradient panelty                           #
#################################################################
def compute_gradient_penalty(netD, real_data, fake_data, device):
    
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = Variable(torch.rand(1),requires_grad=True)

    alpha = alpha.expand(real_data.size()).to(device)
    # print(alpha.shape)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    if device:
        interpolates = interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

    # def compute_gradient_penalty(netD, real_data, fake_data):
    
    # # print "real_data: ", real_data.size(), fake_data.size()
    # alpha = Variable(torch.rand(1),requires_grad=True)

    # # alpha = Tensor(np.random.random((real_data.size(0),1, 1, 1))).to(cuda0)

    # # alpha = Variable(torch.rand(BATCH_SIZE,1,1,1),requires_grad=True)
    # alpha = alpha.expand(real_data.size()).to(cuda0)
    # # print(alpha.shape)
    # interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # if cuda0:
    #     interpolates = interpolates.to(cuda0)
    # interpolates = Variable(interpolates, requires_grad=True)

    # disc_interpolates = netD(interpolates)

    # gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
    #                           grad_outputs=torch.ones(disc_interpolates.size()).to(cuda0),
    #                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    # # gradients = gradients.view(gradients.size(0), -1)

    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # return gradient_penalty