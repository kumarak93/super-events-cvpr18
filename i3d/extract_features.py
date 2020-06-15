import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

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


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='flow', root='../../charades/Charades_v1_flow', split='../data/charades.json', batch_size=1, load_model='./models/flow_charades.pt', save_dir='../../charades_feat_new/Charades_v1_flow_feat'):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('data')

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2) #, final_endpoint='Mixed_5c')
    else:
        i3d = InceptionI3d(400, in_channels=3) #, final_endpoint='Mixed_5c')
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    print('model')

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            #print('data i')
            inputs, labels, name = data
            print(name)
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b,c,t,h,w = inputs.shape
            size_t=200 #1600
            if t > size_t:
                #print('split')
                features = []
                for start in range(1, t-56, size_t):
                    end = min(t-1, start+size_t+56)
                    start = max(1, start-48)
                    ip = torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda() #Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    feat = i3d.extract_features(ip)
                    #print(feat.shape)
                    features.append(feat.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                #print('no split')
                # wrap them in Variable
                inputs = inputs.cuda() #Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                #print(features.shape)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run()
    #run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
