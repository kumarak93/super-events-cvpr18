from __future__ import division
import time
import os
import argparse

import sys
import warnings

from barbar import Bar

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-model_file', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='1')
parser.add_argument('-dataset', type=str, default='charades')

args = parser.parse_args()

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
import json
import sys
#np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

import super_event
#import super_event_sep_xy_t as super_event ########
from apmeter import APMeter

batch_size = 8 #8 #16 #16
if args.dataset == 'multithumos':
    from multithumos_i3d_per_video import MultiThumos as Dataset
    from multithumos_i3d_per_video import mt_collate_fn as collate_fn
    train_split = 'data/multithumos.json'
    test_split = 'data/multithumos.json'
    rgb_root = '/ssd2/thumos/i3d_rgb'
    flow_root = '/ssd2/thumos/i3d_flow'
    classes = 65
elif args.dataset == 'charades':
    from charades_i3d_per_video import MultiThumos as Dataset
    from charades_i3d_per_video import mt_collate_fn as collate_fn
    train_split = 'data/charades.json' #charades_temp
    test_split = 'data/charades.json'
    rgb_root = '/nfs/bigdisk/kumarak/datasets/charades_corrected/Charades_v1_rgb_feat' ##'../charades/Charades_v1_rgb_feat'
    flow_root = '/nfs/bigdisk/kumarak/datasets/charades_corrected/Charades_v1_flow_feat' #'../charades/Charades_v1_flow_feat'
    classes = 157
elif args.dataset == 'ava':
    from ava_i3d_per_video import Ava as Dataset
    from ava_i3d_per_video import ava_collate_fn as collate_fn
    train_split = 'data/ava.json'
    test_split = train_split
    rgb_root = '/ssd2/ava/i3d_rgb'
    flow_root = '/ssd2/ava/i3d_flow'
    classes = 80
    # reduce batchsize as AVA videos are very long
    batch_size = 6
    


def sigmoid(x):
    return 1/(1+np.exp(-x))

def load_data(train_split, val_split, root):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, collate_fn=collate_fn)
        dataloader.root = root
    else:
        
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('data loaded...')
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    since = time.time()

    #best_loss = 10000
    best_map = 0
    for epoch in range(num_epochs):
        
        reg_margin = max(0.3, 1- 0.05*(1+epoch))
        
        print ('Epoch {}/{} : reg_margin {}'.format(epoch, num_epochs - 1, reg_margin))
        print ('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_step(model, gpu, optimizer, dataloader['train'], reg_margin)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'])
            probs.append(prob_val)
            sched.step(val_loss)

            #if val_loss < best_loss:
            if val_map > best_map:
                #best_loss = val_loss
                best_map = val_map
                torch.save(model.state_dict(), model_file) #'models/'+

def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        #print(data[0].shape, data[1].shape, data[2].shape)
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1]/other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def similarity_loss(super_events_1, super_events_2, labels, mask):
    #classes =torch.cat([torch.sum(F.one_hot(torch.unique(torch.where(labels[i]==1)[1]), 
    #    num_classes=labels.shape[-1]), dim=0).view(1,-1) for i in range(0,labels.shape[0])], dim=-1).float() # B C
    
    #labels B T C
    #mask B T
    classes = torch.sum(labels,dim=1) * 100 / torch.sum(mask, dim=1).view(-1,1) # B C
    #print('\n')
    #print(mask.shape[1], torch.sum(mask, dim=1))

    #common = torch.matmul(classes,classes.T)
    #total = (torch.matmul((classes==0)*1.,classes.T)+torch.matmul(classes,(classes.T==0)*1.)+common)
    #gt_sim = (common/(total+1e-6)).clamp_(0, 1)
    u1=classes.repeat(1,classes.shape[0]).view(-1,classes.shape[1])
    u2=classes.repeat(classes.shape[0],1)
    gt_sim = F.cosine_similarity(u1,u2,dim=1).view(classes.shape[0],-1).clamp_(0, 1)
    #print('gt',gt_sim)

    v1=super_events_1.repeat(1,super_events_1.shape[0]).view(-1,super_events_1.shape[1])
    v2=super_events_2.repeat(super_events_2.shape[0],1)
    emb_sim = F.cosine_similarity(v1,v2,dim=1).view(super_events_1.shape[0],-1).clamp_(0, 1)
    #print('emb',emb_sim)

    sim_loss = F.mse_loss(emb_sim, gt_sim)
    return sim_loss


def run_network(model, data, gpu, baseline=False, reg_margin=0):
    # get the inputs
    inputs, mask, labels, other = data
    #print(inputs.shape, mask, labels.shape, other)
    #print(inputs.shape)
    #print(inputs.shape, torch.cuda.max_memory_allocated(device=gpu))
    
    # wrap them in Variable
    inputs = inputs.cuda()
    mask = mask.cuda()
    labels = labels.cuda() # B T C
    #print('m', labels[0,:5,:])
    #toprint=[list(torch.unique(torch.where(labels[i]==1)[1]).cpu().numpy()) for i in range(0,labels.shape[0])]
    #for i in toprint: print(i)
    #print('***\n')
    
    #cls_wts = torch.FloatTensor([1.00]).cuda(gpu)

    # forward
    if not baseline:
        #outputs = model([inputs, torch.sum(mask, 1)])
        #print(outputs.shape)
        outputs, super_events_1, super_events_2 = model([inputs, torch.sum(mask, 1)])
    else:
        #outputs = model(inputs)
        outputs, super_events_1, super_events_2 = model(inputs)
    
    sim_loss = similarity_loss(super_events_1, super_events_2, labels, mask)

    outputs = outputs.squeeze(3).squeeze(3).permute(0,2,1) #.cpu() # remove spatial dims
    ##outputs = outputs.permute(0,2,1) # remove spatial dims
    probs = F.sigmoid(outputs) * mask.unsqueeze(2)
    
    # binary action-prediction loss
    loss = F.binary_cross_entropy_with_logits(outputs, labels, size_average=False)#, weight=cls_wts)

    
    loss = torch.sum(loss) / torch.sum(mask) # mean over valid entries
    
    '''
    if model.training: 
        se1 = model.module.super_event
        se2 = model.module.super_event2
        #reg_xy = torch.sum(se1.sigma_x + se1.sigma_y + se2.sigma_x + se2.sigma_y)
        target = torch.ones(se1.sigma_x.shape).cuda() * reg_margin
        zero = torch.zeros(se1.sigma_x.shape).cuda()
        #print(target)
        reg_xy = F.mse_loss(torch.max(zero, se1.sigma_x-target), zero) + F.mse_loss(torch.max(zero, se1.sigma_y-target), zero) +\
                F.mse_loss(torch.max(zero, se2.sigma_x-target), zero) + F.mse_loss(torch.max(zero, se2.sigma_y-target), zero)
        reg_t = F.mse_loss(torch.max(zero, se1.sigma_t-target), zero) + F.mse_loss(torch.max(zero, se2.sigma_t-target), zero)
        #reg_t = F.mse_loss(se1.sigma_t,target) + F.mse_loss(se2.sigma_t,target) 
        #torch.sum(se1.sigma_t + se2.sigma_t)
        r_const = 10
        #sl:%f 10*sim_loss.item()
        print('\nl:%f rl_xy:%f rl_t:%f rl_xyt:%f margin:%f\n'%(loss.item(), reg_xy.item(), reg_t.item(), 
            (0.1*r_const*reg_t + r_const*reg_xy).item(), reg_margin ))
        #print('sl',sim_loss)
        #loss = loss + 0 * sim_loss
        loss = loss + 0.1*r_const*reg_t + r_const*reg_xy #+ 10*sim_loss
    '''
    # compute accuracy
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    #del inputs
    #del mask
    #del labels
    #print('')
    #print(outputs.shape, torch.cuda.max_memory_allocated(device=gpu))
    #torch.cuda.empty_cache()
    #print(outputs.shape, torch.cuda.max_memory_allocated(device=gpu))
    return outputs, loss, probs, corr/tot
            
                

def train_step(model, gpu, optimizer, dataloader, reg_margin):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    
    # Iterate over data.
    tr_apm = APMeter()
    for data in Bar(dataloader):
        optimizer.zero_grad()
        num_iter += 1
        reg = max(0.4, 0.05 * np.exp(-1*num_iter/500.) + (reg_margin-0.05))
        #if num_iter<200: continue
        
        outputs, loss, probs, err = run_network(model, data, gpu, reg_margin=reg)
        #del outputs
        #print(err, loss)
        
        error += err.item() #data[0]
        tot_loss += loss.item() #data[0]

        loss.backward()
        optimizer.step()
        #print(probs.shape, data[2].shape)
        tr_apm.add(probs.view(-1,probs.shape[-1]).detach().cpu().numpy(), data[2].view(-1,data[2].shape[-1]).cpu().numpy())
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print ('train-{} Loss: {:.4f} MAP: {:.4f}'.format(dataloader.root.split('/')[-1], epoch_loss, tr_apm.value().mean())) #error
    tr_apm.reset()

  

def val_step(model, gpu, dataloader):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}


    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]
        
        outputs, loss, probs, err = run_network(model, data, gpu)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        
        error += err.item() #data[0]
        tot_loss += loss.item() #data[0]
        
        # post-process preds
        outputs = outputs.squeeze()
        probs = probs.squeeze()
        fps = outputs.size()[1]/other[1][0]
        full_probs[other[0][0]] = (probs.data.cpu().numpy().T, fps)
        
        
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    #print ('val-map:', apm.value().mean())
    #apm.reset()
    val_map = apm.value().mean()
    print ('val-{} Loss: {:.4f} MAP: {:.4f}'.format(dataloader.root.split('/')[-1], epoch_loss, val_map)) #error
    #print('mu_x %f, sigma_x %f, mu_t %.10f, sigma_t %f, rho_xy %f'%(model.module.super_event.mu_x[0].item(), model.module.super_event.sigma_x[0].item(), 
    #    model.module.super_event.mu_t[0].item(), model.module.super_event.sigma_t[0].item(), model.module.super_event.rho_xy[0].item()))
    #print ('LR:%f'%lr)
    #print('conv1 %f, fc1 %f'%(model.module.super_event.conv1.weight[0,0,0,0,0].item(), model.module.super_event.fc1.weight[0,0].item()))
    #print('sup_mat %f, per_frame %f'%(model.module.sup_mat[0][0][0].item(), model.module.per_frame.weight[0][0][0][0][0].item()))
    apm.reset()
    return full_probs, epoch_loss, val_map


if __name__ == '__main__':

    if args.mode == 'flow':
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)


    if args.train:
        model = super_event.get_super_event_model(0, classes)
        print(model)
        model = torch.nn.DataParallel(model)
        #model.load_state_dict(torch.load(args.rgb_model_file)) #model = torch.load(args.rgb_model_file)  #################
        criterion = nn.NLLLoss(reduce=False)
    
        lr = 0.1*batch_size/len(datasets['train'])
        print ('LR:%f'%lr)
        #print(model.parameters())
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        
        run([(model,0,dataloaders,optimizer, lr_sched, args.model_file)], criterion, num_epochs=40)

    else:
        print ('Evaluating...')
        rgb_model = torch.load(args.rgb_model_file)
        rgb_model.cuda()
        dataloaders, datasets = load_data('', test_split, rgb_root)
        rgb_results = eval_model(rgb_model, dataloaders['val'], baseline=True)

        flow_model = torch.load(args.flow_model_file)
        flow_model.cuda()
        dataloaders, datasets = load_data('', test_split, flow_root)
        flow_results = eval_model(flow_model, dataloaders['val'], baseline=True)

        rapm = APMeter()
        fapm = APMeter()
        tapm = APMeter()


        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
            rapm.add(sigmoid(o), l)
            fapm.add(sigmoid(flow_results[vid][0]), l)
            if vid in flow_results:
                o2,p2,l2,fps = flow_results[vid]
                o = (o[:o2.shape[0]]*.5+o2*.5)
                p = (p[:p2.shape[0]]*.5+p2*.5)
            tapm.add(sigmoid(o), l)
        print ('rgb MAP:', rapm.value().mean())
        print ('flow MAP:', fapm.value().mean())
        print ('two-stream MAP:', tapm.value().mean())
