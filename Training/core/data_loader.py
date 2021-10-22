"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random
import imageio

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder 
from core.own_data_loader import CustomImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, root_masks, attributes, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.root_masks = root_masks
        self.attributes = attributes
        self.transform = transform
        self.targets = None
    
    def random_attribute(self, fname, size, attributes):
        
        width, height = size
        sample_mask = np.zeros([width//2,height//2,3])
        sel_attributes = []
        
        for i in range (len(attributes)):
  
            if attributes[i] == "eyes":
                sel_attributes.append("l_eye")
                sel_attributes.append("r_eye")
                sel_attributes.append("l_brow")
                sel_attributes.append("r_brow")
            elif attributes[i] == "lip":
                sel_attributes.append("l_lip")
                sel_attributes.append("u_lip")
                sel_attributes.append("mouth")
            else:
                sel_attributes.append(attributes[i])
        
        for i in range (len(sel_attributes)):

            mask_file = fname+"_"+sel_attributes[i]+".png"                
            mask_path = os.path.join(self.root_masks, mask_file)

            if os.path.isfile(mask_path):
                img = imageio.imread(mask_path)
                sample_mask = sample_mask + img
                    
        return sample_mask
    
    def _find_masks(self, fname, size):
        """
        Args:
            index (int): Index

        Returns:
            mask
        """
        fname = str(fname.stem)
        try:
            attributes = self.attributes.split(", ")
            mask_eyes = self.random_attribute(fname, size, [attributes[0]])
            mask_lip = self.random_attribute(fname, size, [attributes[1]]) 
            mask_nose = self.random_attribute(fname, size, [attributes[2]])
            mask_skin = self.random_attribute(fname, size, ["skin"])
        except: 
            pass
        
        mask_eyes = Image.fromarray(np.uint8(mask_eyes))
        mask_lip = Image.fromarray(np.uint8(mask_lip))
        mask_nose = Image.fromarray(np.uint8(mask_nose))
        mask_skin = Image.fromarray(np.uint8(mask_skin))

        return mask_eyes, mask_lip, mask_nose, mask_skin

    def __getitem__(self, index):
        fname = self.samples[index]
        sample = Image.open(fname).convert('RGB')

        mask_eyes, mask_lip, mask_nose, mask_skin = self._find_masks(fname, sample.size)
                  
        if self.transform is not None:
            sample = self.transform(sample)
            mask_eyes = self.transform(mask_eyes)
            mask_lip = self.transform(mask_lip)
            mask_nose = self.transform(mask_nose)
            mask_skin = self.transform(mask_skin)
            
        sample_mask = torch.ones([4, mask_eyes.shape[1], mask_eyes.shape[2]])
        sample_mask[0] = mask_eyes[0]
        sample_mask[1] = mask_lip[0]
        sample_mask[2] = mask_nose[0]
        sample_mask[3] = mask_skin[0]
            
        return sample, sample_mask

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]    
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)
        

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, root_masks, attributes, which='source', 
                     img_size=256, batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    training = True
    if which == 'source':
        dataset = CustomImageFolder(root, root_masks, attributes, training, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, root_masks, attributes, img_size=256, 
                    batch_size=32,imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = DefaultDataset(root, root_masks, attributes, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, root_masks, attributes, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    training = False
    dataset = CustomImageFolder(root, root_masks, attributes, training, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y, m = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, m = next(self.iter)
        return x, y, m

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y, m = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, m_src=m, 
                           y_ref=y_ref, x_ref=x_ref, 
                           x_ref2=x_ref2,z_trg=z_trg, 
                           z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref, m_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y, m_src=m,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y, m=m)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})