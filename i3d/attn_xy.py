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
        self.mu_x = nn.Parameter(torch.FloatTensor(N))
        self.mu_y = nn.Parameter(torch.FloatTensor(N))
        self.sigma_x = nn.Parameter(torch.FloatTensor(N))
        self.sigma_y = nn.Parameter(torch.FloatTensor(N))
        self.mu_x.data.normal_(0,0.5) # 0.5, 0.01
        self.mu_y.data.normal_(0,0.5) # 0.5, 0.01
        self.sigma_x.data.normal_(0,0.0001) #0.0001
        self.sigma_y.data.normal_(0,0.0001) #0.0001


    # on T=64 frame
    def get_filters(self, mu_x, mu_y, sigma_x, sigma_y, width, height):
        sigma_x = 1 * torch.exp(1.5 - 2.0 * sigma_x)
        sigma_y = 1 * torch.exp(1.5 - 2.0 * sigma_y)
        Sigma = torch.stack((sigma_x**2+1e-6, torch.zeros(self.Ni).to(torch.float32).cuda(),
                torch.zeros(self.Ni).to(torch.float32).cuda(), sigma_y**2+1e-6), dim=-1).view(self.Ni, 2, 2) # N 2 2
        Sigma = torch.inverse(Sigma)
        Sigma = Sigma.unsqueeze(1).repeat(1,width*height, 1, 1).view(-1,2,2) # NWH 2 2
        x, y = torch.meshgrid([torch.arange(0,width).to(torch.float32).cuda(),
                            torch.arange(0,height).to(torch.float32).cuda()]) # W H
        mu_x = ((width - 1) * ((mu_x.view(-1) + 1) / 2.0)) # N
        mu_y = ((height - 1) * ((mu_y.view(-1) + 1) / 2.0)) # N
        x = x.repeat(self.Ni,1,1) - mu_x.view(-1,1,1)
        y = y.repeat(self.Ni,1,1) - mu_y.view(-1,1,1)
        var = torch.stack((x, y), dim=-1).view(-1, 1, 2) # NWH 1 2
        f = torch.bmm(var, Sigma)
        f = torch.bmm(f, var.view(-1,2,1)) # BNTWH 1 1
        f = torch.exp(-0.5 * f).view(self.Ni, width, height)
        f = f/(torch.sum(f, (1,2), keepdim=True) + 1e-6)
        return f

    def forward(self, inp):
        video, meta = inp # meta B [nf,st]
        batch, channels, time, width, height = video.size() # time=64//2
        vid = video.unsqueeze(2).repeat(1,1,self.Ni,1,1,1) # B C N T W H
        f = self.get_filters(torch.tanh(self.mu_x), torch.tanh(self.mu_y), torch.sigmoid(self.sigma_x),
                            torch.sigmoid(self.sigma_y), width, height) # N W H
        f = f.view(1, 1, self.Ni, 1, width, height).repeat(batch,channels,1,time,1,1)
        #o = torch.sum(f*vid, dim=2) # B C T W H
        o = torch.mean(torch.sum(f*vid, dim=(4,5)), dim=3) # B C N
        o = o.view(batch, channels*self.Ni)
        return o
