import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

class TSF(nn.Module):

    def __init__(self, N=3, M=5, name=''):
        super(TSF, self).__init__()

        self.N = float(N)
        self.Ni = int(N)
        self.name = name
        
        # B 1024 T 7 7
        self.conv1 = nn.Conv3d(1024, 32, (1,1,1), stride=(1,1,1), padding=(0,0,0)) 
        #self.conv2 = nn.Conv3d(256, 64, (3,1,1), stride=(2,1,1), padding=(1,0,0))
        #self.fc1 = nn.Parameter(torch.Tensor(1, 256, 2*N)) #####
        self.fc11 = nn.Linear(32*32*3*3, 1024) #####
        self.fc12 = nn.Linear(1024, 3*N) #####
        self.fc21 = nn.Linear(32*32*3*3, 1024) #####
        self.fc22 = nn.Linear(1024, 3*N) #####
        '''
        stdv1 =  1./np.sqrt(self.fc1.weight.size(1))
        #stdv2 = 1./np.sqrt(self.fc2.weight.size(1))
        stdv3 = 1./np.sqrt(self.fc3.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv1, stdv1)
        self.fc1.bias.data.uniform_(-stdv1, stdv1)
        #self.fc2.weight.data.uniform_(-stdv2, stdv2)
        #self.fc2.bias.data.uniform_(-stdv2, stdv2)
        self.fc3.weight.data.uniform_(-stdv3, stdv3)
        self.fc3.bias.data.uniform_(-stdv3, stdv3)
        #self.conv1.weight.data.uniform_(-stdv, stdv)
        #self.conv1.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        #torch.nn.init.xavier_uniform_(self.conv2.weight)
        #torch.nn.init.zeros_(self.conv2.bias)
        '''
        sd = 1e-5
        self.fc11.weight.data.uniform_(-sd, sd)
        self.fc11.bias.data.uniform_(-sd, sd)
        self.fc12.weight.data.uniform_(-sd, sd)
        self.fc12.bias.data.uniform_(-sd, sd)
        self.fc21.weight.data.uniform_(-sd, sd)
        self.fc21.bias.data.uniform_(-sd, sd)
        self.fc22.weight.data.uniform_(-sd, sd)
        self.fc22.bias.data.uniform_(-sd, sd)
        #self.fc1.weight.data = torch.zeros(self.fc1.weight.size())
        #self.fc1.bias.data = torch.zeros(self.fc1.bias.size())
        #self.fc3.weight.data = torch.zeros(self.fc3.weight.size())
        #self.fc3.bias.data = torch.zeros(self.fc3.bias.size())
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

        #self.mu_t = nn.Parameter(torch.FloatTensor(N))
        #self.sigma_t = nn.Parameter(torch.FloatTensor(N))
        #self.mu_t.data.normal_(0,0.5)
        #self.sigma_t.data.normal_(0,0.0001) #0.0001  
        self.sigmas = torch.tensor([-0.5,0,0.5]).cuda()
        self.mu_t, self.sigma_t = 0, 0
        self.mu_x, self.sigma_y = 0, 0
        self.mu_y, self.sigma_x = 0, 0
        self.rho_tx, self.rho_ty, self.rho_xy = torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1)
        
        #self.rho_tx = nn.Parameter(torch.FloatTensor(N))
        #self.rho_ty = nn.Parameter(torch.FloatTensor(N))
        #self.rho_xy = nn.Parameter(torch.FloatTensor(N))
        #self.rho_tx.data.normal_(0,0.001)
        #self.rho_ty.data.normal_(0,0.001)                                                                               
        #self.rho_xy.data.normal_(0,0.001)

        self.scale_t = 1 #10
        self.scale_xy = 1 #1

    def get_filters_xyt(self, mu_t, mu_x, mu_y, sigma_t, sigma_x, sigma_y, rho_tx, rho_ty, rho_xy, 
                        length, time, width, height, batch):

        sigma_t = self.scale_t * torch.exp(7.0 * torch.abs(sigma_t) - 1.5) # B N 1
        sigma_x = self.scale_xy * torch.exp(5.0 * torch.abs(sigma_x) - 1.5) # B N 1
        sigma_y = self.scale_xy * torch.exp(5.0 * torch.abs(sigma_y) - 1.5) # B N 1

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
        mu_t = ((length - 1) * (mu_t.reshape(-1) + 1) / 2.0) # BN
        mu_x = ((width - 1) * ((mu_x.reshape(-1) + 1) / 2.0)) # BN
        mu_y = ((height - 1) * ((mu_y.reshape(-1) + 1) / 2.0)) # BN 
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

        attn = F.leaky_relu(self.conv1(video)) #.squeeze(3).squeeze(3).transpose(1,2)  # B 256 T H W
        #attn = F.relu(attn, inplace=True)
        #attn = F.relu(self.conv2(attn))
        #attn_xy = attn.permute(0,2,1,3,4).view(batch*time, 256, width, height) # BT 256 H W
        #attn_xy = F.adaptive_avg_pool2d(attn, (1,1)).view(batch*time,-1) # BT 256
        #attn_xy = self.fc2(attn_xy).view(batch,time,-1).permute(0,2,1) # B 4N T
        #attn_xy = torch.mean(attn, dim=(3,4)).transpose(1,2) # B T 256
        #attn_xy = self.fc2(attn_xy).view(batch,time,-1).transpose(1,2) # B 4N T
        #attn_t = F.adaptive_avg_pool3d(attn, (1,1,1)).view(batch,-1) #.squeeze(3).squeeze(3) # B 256 1
        #attn_t = self.fc1(attn_t) # B 2N 1
        
        #attn = F.adaptive_avg_pool3d(attn, (4,3,3)).view(batch,-1) # B 256*8*3*3
        attn = F.adaptive_avg_pool3d(attn, (32,3,3)).view(batch,-1) # B 64*32*7*7
        attn1 = F.leaky_relu(self.fc11(attn)) #.view(batch,-1) # B 6N
        attn1 = F.sigmoid(1e-3 * self.fc12(attn1))
        attn2 = F.leaky_relu(self.fc21(attn))
        attn2 = F.sigmoid(1e-3 * self.fc22(attn2))

        self.mu_t, self.sigma_t = attn1[:,:self.Ni], self.sigmas.view(1,-1).repeat(batch,1) #* attn2[:,:self.Ni] # B N 1
        self.mu_x, self.sigma_x = attn1[:,self.Ni:2*self.Ni], self.sigmas.view(1,-1).repeat(batch,1) #1 * attn2[:,self.Ni:2*self.Ni] # B N 1
        self.mu_y, self.sigma_y = attn1[:,2*self.Ni:3*self.Ni], self.sigmas.view(1,-1).repeat(batch,1) #1 * attn2[:,2*self.Ni:3*self.Ni] # B N 1
        #self.rho_tx = F.tanh(attn[:,6*self.Ni:7*self.Ni])
        #self.rho_ty = F.tanh(attn[:,7*self.Ni:8*self.Ni])
        #self.rho_xy = F.tanh(attn[:,8*self.Ni:9*self.Ni])
        #print(self.training)
        if self.name == 'se_1' and self.training:
            print('\n',time)
            print(self.name,'t',self.mu_t[0].detach().cpu().numpy(), self.sigma_t[0].detach().cpu().numpy())
            print(self.name,'x',self.mu_x[0].detach().cpu().numpy(), self.sigma_x[0].detach().cpu().numpy())
            print(self.name,'y',self.mu_y[0].detach().cpu().numpy(), self.sigma_y[0].detach().cpu().numpy())

        #fg = self.get_filters(self.mu_t, self.mu_x, self.mu_y, self.sigma_t, self.sigma_x, self.sigma_y,
        #                    length.view(batch,1).repeat(1,self.Ni).view(-1), time, width, height, batch) # B N TWH
        #fg = fg.unsqueeze(1).repeat(1, channels, 1, 1) # B D N TWH
        #fg = fg.view(batch*channels, self.Ni, -1) # BD N TWH

        f_xyt = self.get_filters_xyt(self.mu_t, self.mu_x, self.mu_y,self.sigma_t, 
            self.sigma_x, self.sigma_y,F.tanh(self.rho_tx), F.tanh(self.rho_ty), F.tanh(self.rho_xy),
            length.view(batch,1).repeat(1,self.Ni).view(-1), time, width, height, batch) # B N TWH
        
        f_xyt = f_xyt.unsqueeze(1).repeat(1,channels,1,1).view(batch*channels,self.Ni,-1)
        #f_xyt = f_xyt.repeat(channels, 1, 1) # BD N TWH

        video = F.dropout(video, p=0.7)
        
        vid = video.view(batch*channels, -1, 1) # BD TWH 1

        o = torch.bmm(f_xyt, vid) # BD N 1
        del f_xyt
        del vid
        o = o.view(batch, channels*self.Ni)
        return o



