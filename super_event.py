import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import temporal_structure_filter as tsf
#import tsf_sep_xy_t_v2 as tsf
#import tsf_sep_xy_t_v3 as tsf
#import self_attn_xyt as tsf
#import self_attn_xyt_gauss as tsf
#import self_attn_xyt_gauss_combined_v6 as tsf
#import static_attn_xyt_gauss_combined as tsf
#import self_attn_xyt_gauss_combined_v3_constInit as tsf
#import static_attn_xy_t_gauss_sep_constInit as tsf

class SuperEvent(nn.Module):
    def __init__(self, classes=65):
        super(SuperEvent, self).__init__()

        self.classes = classes
        N = 3
        self.dropout = nn.Dropout(0.7)
        self.add_module('d', self.dropout)

        self.super_event = tsf.TSF(N,N, name='se_1').cuda()
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(N,N, name='se_2').cuda()
        self.add_module('sup2', self.super_event2)

        #self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7],stride=(1, 1, 1))

        # we have 2xD*3
        # we want to learn a per-class weighting
        # to take 2xD*3 to D*3
        self.cls_wts = nn.Parameter(torch.Tensor(classes))

        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024*N *1)) #####
        stdv = 1./np.sqrt(self.sup_mat.size(1)) #(1024+1024)
        self.sup_mat.data.uniform_(-stdv, stdv)

        #self.sim_wts_1 = nn.Parameter(torch.Tensor(2,6))
        #self.sim_wts_2 = nn.Parameter(torch.Tensor(2,6))
        #sim_sd = 1./np.sqrt(self.sim_wts_1.size(1))
        #self.sim_wts_1.data.uniform_(-sim_sd, sim_sd)
        #self.sim_wts_2.data.uniform_(-sim_sd, sim_sd)

        self.per_frame = nn.Conv3d(1024, classes, (1,1,1)) #### (1,1,1)/(1,7,7)
        torch.nn.init.xavier_uniform_(self.per_frame.weight) #self.per_frame.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.zeros_(self.per_frame.bias) #self.per_frame.bias.data.uniform_(-stdv, stdv)
        self.add_module('pf', self.per_frame)

    def forward(self, inp):
        #if len(inp[0].size())!=5:
        #    inp = (inp[0].unsqueeze(0), inp[1])
        batch, channels, time, width, height = inp[0].size()
        #inp[0] = self.avg_pool(inp[0])
        inp[0] = self.dropout(inp[0])
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        #print(inp[0].size())
        se1 = self.super_event(inp).squeeze() # B DN
        se2 = self.super_event2(inp).squeeze()
        super_event = self.dropout(torch.stack([se1, se2], dim=dim))
        #super_event = torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim)
        if val:
            super_event = super_event.unsqueeze(0)
        # we have B x 2 x D*3
        # we want B x C x D*3

        #super_event_out_1 = super_event.view(super_event.shape[0],-1)
        #super_event_out_2 = super_event.view(super_event.shape[0],-1)

        #se = torch.cat((se1.view(batch,channels,-1),se2.view(batch,channels,-1)), dim=-1).view(batch*channels,1,-1) # BD 1 2N
        #se_w1 = self.sim_wts_1.repeat(batch*channels,1,3).view(batch*channels,6,6)
        #se_w2 = self.sim_wts_2.repeat(batch*channels,1,3).view(batch*channels,6,6)
        #super_event_out_1 = torch.bmm(se, se_w1).view(batch,-1)
        #super_event_out_2 = torch.bmm(se, se_w2).view(batch,-1)

        #super_event_out_1 = super_event.view(batch, 2, channels, 3).transpose(1,2).reshape(batch,channels,-1)
        #super_event_out_1 = torch.bmm(super_event_out_1, self.sim_wts_1.repeat(batch,1,1)).view(batch,-1)

        #super_event = super_event_out_1.view(batch, channels, 2, 3).transpose(1,2).reshape(batch,2,channels*3)

        #super_event_out_2 = super_event.view(batch, 2, channels, 3).transpose(1,2).reshape(batch,channels,-1)
        #super_event_out_2 = torch.bmm(super_event_out_2, self.sim_wts_2.repeat(batch,1,1)).view(batch,-1)


        #print(super_event.size())
        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        #print cls_wts.expand(inp[0].size()[0], -1, -1).size(), super_event.size()
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)
        del cls_wts
        #print(super_event.size())
        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)
        #super_event = self.sup_mat(super_event.view(-1, 1024)).view(-1, self.classes)

        super_event = super_event.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        #print(super_event.size())

        #inp[0] = self.dropout(inp[0])

        #cls = self.per_frame(torch.mean(inp[0], dim=(3,4), keepdim=True)) ##### 1D
        #cls = self.per_frame(self.avg_pool(inp[0])) ##### 1D
        cls = self.per_frame(inp[0])

        #return torch.cat((super_event,cls),dim=0), super_event_out_1, super_event_out_2
        #return super_event+cls, super_event_out_1, super_event_out_2
        return cls #, super_event_out_1, super_event_out_2



def get_super_event_model(gpu, classes=65):
    model = SuperEvent(classes)
    return model.cuda()
