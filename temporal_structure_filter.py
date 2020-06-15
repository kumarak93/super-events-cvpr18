import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class TSF(nn.Module):

    def __init__(self, N=3, name=''):
        super(TSF, self).__init__()

        self.N = float(N)
        self.Ni = int(N)

        self.mu_t = nn.Parameter(torch.FloatTensor(N))
        self.sigma_t = nn.Parameter(torch.FloatTensor(N))

        self.mu_t.data.normal_(0,0.5)
        self.sigma_t.data.normal_(0,0.0001) #0.0001


    def get_filters(self, mu_t, sigma_t, length, time, batch):

        sigma_t = 1 * torch.exp(1.5 - 2.0 * torch.abs(sigma_t))
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(self.Ni*batch, 1) - ((length - 1) * (mu_t.repeat(batch) + 1) / 2.0).view(-1,1)
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(batch,time) + 1e-6)
        f = torch.exp(-f)
        f = f / (torch.sum(f, 1).view(-1,1)+1e-6)
        f = f.view(-1, self.Ni, time)
        return f

    def forward(self, inp):
        video, length = inp
        batch, channels, time = video.squeeze(3).squeeze(3).size()
        # vid is (B x C x T)
        vid = video.view(batch*channels, time, 1).unsqueeze(2)
        # f is (B x T x N)
        f = self.get_filters(torch.tanh(self.mu_t), torch.tanh(self.sigma_t), length.view(batch,1).repeat(1,self.Ni).view(-1), time, batch)
        # repeat over channels
        fout = f # B N T
        f = f.unsqueeze(1).repeat(1, channels, 1, 1)
        f = f.view(batch*channels, self.Ni, time)

        # o is (B x C x N)
        o = torch.bmm(f, vid.squeeze(2))
        del f
        del vid
        o = o.view(batch, channels*self.Ni)#.unsqueeze(3).unsqueeze(3)
        return o, fout


