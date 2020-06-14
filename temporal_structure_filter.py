import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

'''
class TSF(nn.Module):

    def __init__(self, N=3):
        super(TSF, self).__init__()

        self.N = float(N)
        self.Ni = int(N)

        # create parameteres for center and delta of this super event
        self.center = nn.Parameter(torch.FloatTensor(N))
        self.delta = nn.Parameter(torch.FloatTensor(N))
        self.gamma = nn.Parameter(torch.FloatTensor(N))

        # init them around 0

        self.center.data.normal_(0,0.5)
        self.delta.data.normal_(0,0.01)
        self.gamma.data.normal_(0, 0.0001)


    def get_filters(self, delta, gamma, center, length, time):
        """
            delta (batch,) in [-1, 1]
            center (batch,) in [-1, 1]
            gamma (batch,) in [-1, 1]
            length (batch,) of ints
        """

        # scale to length of videos
        centers = (length - 1) * (center + 1) / 2.0
        deltas = length * (1.0 - torch.abs(delta))

        gammas = torch.exp(1.5 - 2.0 * torch.abs(gamma))
        
        a = Variable(torch.zeros(self.Ni))
        a = a.cuda()
        
        # stride and center
        a = deltas[:, None] * a[None, :]
        a = centers[:, None] + a

        b = Variable(torch.arange(0, time))
        b = b.cuda()
        
        f = b - a[:, :, None]
        f = f / gammas[:, None, None]
        
        f = f ** 2.0
        f += 1.0
        f = np.pi * gammas[:, None, None] * f
        f = 1.0/f
        f = f/(torch.sum(f, dim=2) + 1e-6)[:,:,None]

        f = f[:,0,:].contiguous()

        f = f.view(-1, self.Ni, time)
        
        return f

    def forward(self, inp):
        video, length = inp
        batch, channels, time = video.squeeze(3).squeeze(3).size()
        # vid is (B x C x T)
        vid = video.view(batch*channels, time, 1).unsqueeze(2)
        # f is (B x T x N)
        f = self.get_filters(torch.tanh(self.delta).repeat(batch), torch.tanh(self.gamma).repeat(batch), torch.tanh(self.center.repeat(batch)), length.view(batch,1).repeat(1,self.Ni).view(-1), time)
        # repeat over channels
        f = f.unsqueeze(1).repeat(1, channels, 1, 1)
        f = f.view(batch*channels, self.Ni, time)

        # o is (B x C x N)
        o = torch.bmm(f, vid.squeeze(2))
        del f
        del vid
        o = o.view(batch, channels*self.Ni)#.unsqueeze(3).unsqueeze(3)
        return o

'''
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
        #print("sigma_t",sigma_t)
        t = torch.arange(0,time).to(torch.float32).cuda()
        t = t.repeat(self.Ni*batch, 1) - ((length - 1) * (mu_t.repeat(batch) + 1) / 2.0).view(-1,1)
        #print('t',t)
        f = t**2 / (2 * (sigma_t**2).view(self.Ni,1).repeat(batch,time) + 1e-6)
        #print('f1',f)
        f = torch.exp(-f)
        #print('f2',f)#print(f.shape, torch.sum(f, 1).shape)
        f = f / (torch.sum(f, 1).view(-1,1)+1e-6)
        #print('f3',f)
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


