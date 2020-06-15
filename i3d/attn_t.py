import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class AttnLayer(nn.Module):

    def __init__(self, N=3, name=''):
        super(AttnLayer, self).__init__()

        self.N = float(N)
        self.Ni = int(N)
        self.mu_t = nn.Parameter(torch.FloatTensor(N))
        self.sigma_t = nn.Parameter(torch.FloatTensor(N))
        self.mu_t.data.normal_(0,0.5)
        self.sigma_t.data.normal_(0,0.0001) #0.0001


    def get_filters(self, mu_t, sigma_t, time):

        sigma_t = 1 * torch.exp(1.5 - 2.0 * sigma_t)
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(self.Ni, 1) - ((time - 1) * (mu_t + 1) / 2.0).view(-1,1) # N T
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(1,time) + 1e-16)
        f = torch.exp(-f)
        f = f / (torch.sum(f, dim=1, keepdim=True) + 1e-16)
        f = f.view(self.Ni, time)
        return f

    def forward(self, inp):

        video = inp #video, length = inp
        batch, channels, time, width, height = video.size()
        vid = video.unsqueeze(2).repeat(1,1,self.Ni,time,1,1) # B C N T W H
        f = self.get_filters(torch.tanh(self.mu_t), torch.sigmoid(self.sigma_t), time)
        f = f.view(1,1,self.Ni,time,1,1) # 1 1 N T 1 1
        o = torch.sum(f*vid, dim=2) # B C T W H
        return o


