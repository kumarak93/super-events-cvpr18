import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class TSF(nn.Module):

    def __init__(self, N=3, M=5, name=''):
        super(TSF, self).__init__()

        self.N = float(N)
        self.Ni = int(N)
        self.name = name

        self.mu_t = nn.Parameter(torch.FloatTensor(N))
        self.mu_x = nn.Parameter(torch.FloatTensor(N))
        self.mu_y = nn.Parameter(torch.FloatTensor(N))
        self.sigma_t = nn.Parameter(torch.FloatTensor(N))
        self.sigma_x = nn.Parameter(torch.FloatTensor(N))
        self.sigma_y = nn.Parameter(torch.FloatTensor(N))
        #self.rho_xy = nn.Parameter(torch.FloatTensor(M))
        self.rho_tx, self.rho_ty, self.rho_xy = torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1)

        self.mu_t.data.normal_(0,0.5)
        self.mu_x.data.normal_(0,0.5)
        self.mu_y.data.normal_(0,0.5)
        self.sigma_t.data.normal_(0,0.0001) #0.0001        
        self.sigma_x.data.normal_(0,0.0001) #0.0001                                                                                  
        self.sigma_y.data.normal_(0,0.0001) #0.0001                                                                                  
        #self.rho_xy.data.normal_(0,0.0001) #0.0001

        self.scale_t = 1 #10
        self.scale_xy = 1 #0.5

    def get_filters_xyt(self, mu_t, mu_x, mu_y, sigma_t, sigma_x, sigma_y, rho_tx, rho_ty, rho_xy, 
                        length, time, width, height, batch):

        sigma_t = self.scale_t * torch.exp(1.5 - 2.0 * torch.abs(sigma_t)) # B N 1
        sigma_x = self.scale_xy * torch.exp(1.5 - 2.0 * torch.abs(sigma_x)) # B N 1
        sigma_y = self.scale_xy * torch.exp(1.5 - 2.0 * torch.abs(sigma_y)) # B N 1

        #print(sigma_t[0],sigma_x[0],sigma_y[0])
        Sigma = torch.stack((sigma_t**2+1e-6, torch.zeros(batch,self.Ni).to(torch.float32).cuda(), torch.zeros(batch,self.Ni).to(torch.float32).cuda(),
            torch.zeros(batch,self.Ni).to(torch.float32).cuda(), sigma_x**2+1e-6, torch.zeros(batch,self.Ni).to(torch.float32).cuda(),
            torch.zeros(batch,self.Ni).to(torch.float32).cuda(), torch.zeros(batch,self.Ni).to(torch.float32).cuda(), sigma_y**2+1e-6), dim=-1).view(batch*self.Ni, 3, 3) # BN 3 3


        #Sigma = torch.stack((sigma_t**2+1e-6, sigma_t*sigma_x*rho_tx, sigma_t*sigma_y*rho_ty,
        #    sigma_t*sigma_x*rho_tx, sigma_x**2+1e-6, sigma_x*sigma_y*rho_xy,          
        #    sigma_t*sigma_y*rho_ty, sigma_x*sigma_y*rho_xy, sigma_y**2+1e-6), dim=-1).view(batch*self.Ni, 3, 3) # BN 3 3
        
        
        #print('s', Sigma[0])
        Sigma = torch.inverse(Sigma)
        #print('s-1', Sigma[0])
        Sigma = Sigma.unsqueeze(1).repeat(1,time*width*height, 1, 1).view(-1,3,3) # BNTWH 3 3

        t, x, y = torch.meshgrid([torch.arange(0,time).to(torch.float32).cuda(),
            torch.arange(0,width).to(torch.float32).cuda(),
            torch.arange(0,height).to(torch.float32).cuda()]) # T W H 

        #print(mu_t.shape, length.shape)
        mu_t = ((length - 1) * (mu_t.view(-1) + 1) / 2.0) # BN
        mu_x = ((width - 1) * ((mu_x.view(-1) + 1) / 2.0)) # BN
        mu_y = ((height - 1) * ((mu_y.view(-1) + 1) / 2.0)) # BN 
        t = t.repeat(batch*self.Ni,1,1,1) - mu_t.view(-1,1,1,1) # BN T W H
        x = x.repeat(batch*self.Ni,1,1,1) - mu_x.view(-1,1,1,1)
        y = y.repeat(batch*self.Ni,1,1,1) - mu_y.view(-1,1,1,1)
        var = torch.stack((t, x, y), dim=-1).view(-1, 1, 3) # BNTWH 1 3

        f = torch.bmm(var, Sigma)
        #print(f)
        f = torch.bmm(f, var.view(-1,3,1)) # BNTWH 1 1
        #print('f',f)
        f = torch.exp(-0.5 * f).view(batch*self.Ni, time, width, height)
        f = f/(torch.sum(f, (1,2,3), keepdim=True) + 1e-6)

        f = f.view(-1, self.Ni, time*width*height)
        #f=torch.clamp(f, 1e-6, 1)
        #print('f', f[0].sum(),f[:,0,200:210])
        return f  


    def forward(self, inp, lamb=1):
        video, length = inp
        batch, channels, time, width, height = video.size()


        mu_t, sigma_t = F.tanh(self.mu_t).view(1,-1).repeat(batch,1), F.tanh(self.sigma_t).view(1,-1).repeat(batch,1) # B N 1
        mu_x, sigma_x = F.tanh(self.mu_x).view(1,-1).repeat(batch,1), F.tanh(self.sigma_x).view(1,-1).repeat(batch,1) # B N 1
        mu_y, sigma_y = F.tanh(self.mu_y).view(1,-1).repeat(batch,1), F.tanh(self.sigma_y).view(1,-1).repeat(batch,1) # B N 1
        if self.name == 'se_1':
            print('\n',time)
            print(self.name,'t',mu_t[0].detach().cpu().numpy(), sigma_t[0].detach().cpu().numpy())
            print(self.name,'x',mu_x[0].detach().cpu().numpy(), sigma_x[0].detach().cpu().numpy())
            print(self.name,'y',mu_y[0].detach().cpu().numpy(), sigma_y[0].detach().cpu().numpy())
        
        #fg = self.get_filters(self.mu_t, self.mu_x, self.mu_y, self.sigma_t, self.sigma_x, self.sigma_y,
        #                    length.view(batch,1).repeat(1,self.Ni).view(-1), time, width, height, batch) # B N TWH
        #fg = fg.unsqueeze(1).repeat(1, channels, 1, 1) # B D N TWH
        #fg = fg.view(batch*channels, self.Ni, -1) # BD N TWH

        f_xyt = self.get_filters_xyt(mu_t, mu_x, mu_y, sigma_t, 
            sigma_x, sigma_y, F.tanh(self.rho_tx), F.tanh(self.rho_ty), F.tanh(self.rho_xy),
            length.view(batch,1).repeat(1,self.Ni).view(-1), time, width, height, batch) # B N TWH
        
        f_xyt = f_xyt.unsqueeze(1).repeat(1,channels,1,1).view(batch*channels,self.Ni,-1)
        #f_xyt = f_xyt.repeat(channels, 1, 1) # BD N TWH

        #video = F.dropout(video, p=0.7)
        
        vid = video.view(batch*channels, -1, 1) # BD TWH 1

        o = torch.bmm(f_xyt, vid) # BD N 1
        del f_xyt
        del vid
        o = o.view(batch, channels*self.Ni)
        return o


