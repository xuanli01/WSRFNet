#!/usr/bin/python3
# coding=utf-8
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import mmcv
from PIL import Image

########################### Data Preprocess ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        return image, mask

########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        if self.kwargs['eval_dataset'] == 'IDRID':
            self.mean = np.array([[[116.513, 56.437, 16.309]]])
            self.std = np.array([[[80.206, 41.232, 13.293]]])
            self.H = 960
            self.W = 1440
        elif self.kwargs['eval_dataset'] == 'DDR':
            self.mean = np.array([[[81.205, 50.636, 21.216]]])
            self.std = np.array([[[76.252, 48.798, 21.625]]])     
            self.H = 1024
            self.W = 1024
        else:
            print('The available datasets include IDRID and DDR')
            assert self.kwargs['eval_dataset'] == 'IDRID' or self.kwargs['eval_dataset'] == 'DDR'
                   
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.resize = Resize(self.cfg.H, self.cfg.W)
        self.totensor = ToTensor()

        self.file_client_args ={'backend': 'disk'}
        self.imdecode_backend = 'pillow'

        self.samples = os.listdir(os.path.join(cfg.datapath, 'image'))

    def __getitem__(self, idx):
        name  = self.samples[idx][:-4]
        image = Image.open(os.path.join(self.cfg.datapath, 'image', name + '.jpg')).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        
        filename = os.path.join(self.cfg.datapath, 'annotations', name + '.png')      
        file_client = mmcv.FileClient(**self.file_client_args)    
        img_bytes = file_client.get(filename)
        mask = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
            
        shape = mask.shape
        image = self.resize(image)
        image = self.normalize(image)
        image, mask = self.totensor(image, mask)
        return image, mask, shape, name

    def __len__(self):
        return len(self.samples)
