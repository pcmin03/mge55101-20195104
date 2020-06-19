import numpy as np
import torch.nn as nn
import torch
import math

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


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

class create_perceptual_loss(nn.Module):
    def __init__(self, vgg_loss,loss):
        super(create_perceptual_loss, self).__init__()

        self.vgg_loss = vgg_loss
        self.loss = loss
    def forward(self,input_,predict):
        input_features = self.vgg_loss(input_)
        predict_features = self.vgg_loss(predict)
        loss_value = 0
        for i in range(int(len(predict_features))):
            loss_value += self.loss(input_features[i],predict_features[i])
        return loss_value

