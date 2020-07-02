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
        self.mu_t.data.normal_(0,0.5) # 0.5, 0.01
        self.sigma_t.data.normal_(0,0.0001) #0.0001




    '''
    def get_filters(self, mu_t, sigma_t, length, time, batch):
        sigma_t = 1 * torch.exp(1.5 - 2.0 * sigma_t)
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(batch*self.Ni, 1) - ((length - 1) * (mu_t.repeat(batch) + 1) / 2.0).view(-1,1) # BN Tf
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(batch,time) + 1e-16)
        f = torch.exp(-f)
        f = f / (torch.sum(f, dim=1, keepdim=True) + 1e-16)
        f = f.view(-1, self.Ni, time)
        return f

    def forward(self, inp):
        # adjust length and mask --- this version works for only val batch=1
        video, meta = inp # meta B [nf,st]
        #meta = (meta - 1)//2 + 1 # account for pooling -- change -- not universally correct
        #frames, starts = meta[:,0], meta[:,1]
        #frames, starts = frames//2, starts//2 # account for pooling -- change
        batch, channels, time, width, height = video.size() # time=64//2
        if self.training:
            scale = 64//time;
        else:
            scale = meta[0]//time
        meta = meta//scale
        frames, starts = meta[:,0], meta[:,1]

        vid = video.unsqueeze(2).repeat(1,1,self.Ni,1,1,1) # B C N T W H
        f = self.get_filters(torch.tanh(self.mu_t), torch.sigmoid(self.sigma_t),
                                frames.view(batch,1).repeat(1,self.Ni).view(-1), max(frames).item(), batch) # B N Tf
        if self.training: # try to do this without for loops
            f_arr = []
            for i in range(batch):
                st_i = starts[i].item()
                f_arr.append(f[i, :, st_i:st_i+time])
            f = torch.stack(f_arr, dim=0) # B N T
        f = f[:,:,:time].view(batch,1,self.Ni,time,1,1) # B 1 N T 1 1
        #print(vid.shape, f.shape)
        #o = torch.sum(f*vid, dim=2) # B C T W H
        o = torch.mean(torch.sum(f*vid, dim=3), dim=(3,4)) # B C N
        o = o.view(batch, channels*self.Ni)
        return o
    '''

    # on T=64 frame
    def get_filters(self, mu_t, sigma_t, time):
        sigma_t = 1 * torch.exp(1.5 - 2.0 * sigma_t)
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(self.Ni, 1) - ((time - 1) * (mu_t + 1) / 2.0).view(-1,1) # N Tf
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(1,time) + 1e-16)
        f = torch.exp(-f)
        f = f / (torch.sum(f, dim=1, keepdim=True) + 1e-16)
        f = f.view(self.Ni, time)
        return f

    def forward(self, inp):
        video, meta = inp # meta B [nf,st]
        batch, channels, time, width, height = video.size() # time=64//2
        vid = video.unsqueeze(2).repeat(1,1,self.Ni,1,1,1) # B C N T W H
        f = self.get_filters(torch.tanh(self.mu_t), torch.sigmoid(self.sigma_t), time) # N T
        f = f.view(1, 1, self.Ni, time, 1, 1).repeat(batch,channels,1,1,1,1)
        #o = torch.sum(f*vid, dim=2) # B C T W H
        o = torch.mean(torch.sum(f*vid, dim=3), dim=(3,4)) # B C N
        o = o.view(batch, channels*self.Ni)
        return o
    
