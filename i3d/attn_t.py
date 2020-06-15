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


    def get_filters(self, mu_t, sigma_t, length, time):

        sigma_t = 1 * torch.exp(1.5 - 2.0 * sigma_t)
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(batch*self.Ni, 1) - ((length - 1) * (mu_t.repeat(batch) + 1) / 2.0).view(-1,1) # B N Tf
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(batch,time) + 1e-16)
        f = torch.exp(-f)
        f = f / (torch.sum(f, dim=1, keepdim=True) + 1e-16)
        f = f.view(-1, self.Ni, time)
        return f

    def forward(self, inp):
        # adjust length and mask
        video, meta = inp # meta B [nf,st]
        frames, starts = meta[:0], meta[:1]
        frames = frames // 2 # account for pooling -- change

        batch, channels, time, width, height = video.size() # time=64
        vid = video.unsqueeze(2).repeat(1,1,self.Ni,1,1,1) # B C N T W H
        f = self.get_filters(torch.tanh(self.mu_t), torch.sigmoid(self.sigma_t), 
                                frames.view(batch,1).repeat(1,self.Ni).view(-1), max(frames).item()) # B N Tf
        if self.training: # try to do this without for loops
            f_arr = []
            for i in range(batch):
                f_arr.append(f[i, :, starts[i].item():starts[i].item()+time])
            f = torch.stack(f_arr, dim=0) # B N T
        f = f.view(batch,1,self.Ni,time,1,1) # B 1 N T 1 1
        #print(vid.shape, f.shape)
        o = torch.sum(f*vid, dim=2) # B C T W H
        return o


