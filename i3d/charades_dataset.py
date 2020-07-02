import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num, i3d_in):
  frames = []
  file = os.path.join(i3d_in, vid+'.npy')
  if os.path.exists(file):
      frames = np.load(file, allow_pickle=True)[start-1:start-1+num, ...]
      frames = np.asarray((frames/255.)*2 - 1, dtype=np.float32)
      #print('file exists: %s'%vid, frames.shape)
      return frames
  else:
      for i in range(start, start+num):
        img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        #img = (img/255.)*2 - 1
        frames.append(img)
      frames = np.asarray(frames, dtype=np.uint8)
      #np.save(file, frames)
      frames = np.asarray((frames/255.)*2 - 1, dtype=np.float32)
      #print('file does not exists: %s'%vid, frames.shape)
      return frames

def load_flow_frames(image_dir, vid, start, num, i3d_in):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)

    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, i3d_in, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    pre_data_file = split_file[:-5]+'_'+split+'labeldata_subset.npy'
    #pre_data_file = split_file[:-5]+'_'+split+'labeldata_64x4.npy'
    #a=[]
    if os.path.exists(pre_data_file):
        print('{} exists'.format(pre_data_file))
        dataset = np.load(pre_data_file, allow_pickle=True)
        #for dat in dataset:
        #    if dat[3] >= 64*4+2:
        #        a.append(dat)
        #np.save(pre_data_file2, a)

    else:
        print('{} does not exist'.format(pre_data_file))
        i = 0
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue

            #if not os.path.exists(os.path.join(root, vid)):
            file = os.path.join(i3d_in, vid+'.npy')
            if not os.path.exists(file):
                continue
            #num_frames = len(os.listdir(os.path.join(root, vid)))
            num_frames = np.load(file, allow_pickle=True).shape[0]
            if mode == 'flow':
                num_frames = num_frames//2

            if num_frames < 64*1+2:
                continue

            label = np.zeros((num_classes,num_frames), np.float32)

            fps = num_frames/data[vid]['duration']
            for ann in data[vid]['actions']:
                #label[ann[0], int(np.ceil(ann[1]*fps)):min(int(np.ceil(ann[2]*fps)),num_frames)] = 1
                for fr in range(0,num_frames,1):
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], fr] = 1 # binary classification
            dataset.append((vid, label, data[vid]['duration'], num_frames))
            i += 1
            print(i, vid)
        np.save(pre_data_file, dataset)

    print('dataset size:%d'%len(dataset))
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, i3d_in=''):

        if split == 'training':
            limit = 1000
        elif split == 'testing':
            limit = 500
        self.data = make_dataset(split_file, split, root, mode, i3d_in)[:limit]
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.i3d_in = i3d_in

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        start_f = random.randint(1,nf-(64*1+1))

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64*1, self.i3d_in)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64*1, self.i3d_in)
        label = label[:, start_f:start_f+64*1]

        imgs = self.transforms(imgs)

        meta = np.array([nf,start_f])

        return video_to_tensor(imgs), torch.from_numpy(label), torch.from_numpy(meta), vid

    def __len__(self):
        return len(self.data)
