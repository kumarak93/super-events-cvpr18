import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
#parser.add_argument('-save_model', type=str)
parser.add_argument('-root', default='/nfs/bigdisk/kumarak/datasets/charades/Charades_v1_rgb', type=str)
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
from torchsummary import summary

import numpy as np
from barbar import Bar
from apmeter import APMeter

from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset
from charades_dataset_full import Charades as Dataset_Full

import warnings
warnings.filterwarnings("ignore")

def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/nfs/bigdisk/kumarak/datasets/charades/Charades_v1_rgb', train_split='../data/charades.json', batch_size=8): #, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    val_dataset = Dataset_Full(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('datasets created')
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(157, in_channels=2) #400 for imagenet_kinetics
        state = i3d.state_dict()
        state.update(torch.load('models/flow_charades.pt'))
        i3d.load_state_dict(state)
        save_model = 'models/flow_temp_'
    else:
        i3d = InceptionI3d(157, in_channels=3)
        state = i3d.state_dict()
        state.update(torch.load('models/rgb_charades.pt'))
        i3d.load_state_dict(state)
        save_model = 'models/rgb_temp_attnT_'
    #i3d.replace_logits(157)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()

    i3d.freeze('Mixed_5c')
    
    for name, param in i3d.named_parameters():
        if param.requires_grad:print('updating: {}'.format(name))
        #else:print('frozen: {}'.format(name))
    
    #summary(i3d, (3, 64, 224, 224))
    i3d = nn.DataParallel(i3d)
    print('model loaded')

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    val_apm = APMeter()
    tr_apm = APMeter()
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            print(phase)
            for data in Bar(dataloaders[phase]):
                num_iter += 1
                #print(num_iter)
                # get the inputs
                inputs, labels, meta = data
                #print(meta)
                #print(inputs.shape, labels.shape) #(B Ch=3 T=64 H=224 W=224) (B C=157 T)

                # wrap them in Variable
                inputs = Variable(inputs.cuda()) # B 3 T W H
                t = inputs.size(2)
                labels = Variable(labels.cuda()) # B C T

                per_frame_logits = i3d([inputs, meta]) # B C T//16 
                
                # upsample to input size
                #per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                labels = F.interpolate(labels, size=per_frame_logits.shape[2])
                
                probs = F.sigmoid(per_frame_logits)
                #print(labels.shape, probs.shape)

                if phase == 'train':
                    for b in range(labels.shape[0]):
                        tr_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())
                else:
                    for b in range(labels.shape[0]):
                        val_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item() #data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = 0 * F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item() #data[0]

                loss = 2 * (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item() #data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        tr_map = tr_apm.value().mean()
                        #i3d.module.get_attn_para() #### print mu, sigma
                        print ('{} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(phase, 
                            steps, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10, tr_map))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                    if steps % 100 == 0:
                        tr_apm.reset()
            if phase == 'val':
                val_map = val_apm.value().mean()
                val_apm.reset()
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(phase, 
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map)) 
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root) #, save_model=args.save_model)
