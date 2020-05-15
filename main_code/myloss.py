import numpy as np
import torch.nn as nn
import torch
import math

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