# coding:utf-8

import torch
from torchvision import transforms
from torch.utils import data

import glob
from skimage import io, transform
import random

class Nuclei(data.Dataset):
    def __init__(self, config, pahse='train', ings_trans=None, masks_trans=None):
        '''
        __init__() just get img and mask's path
        '''
        assert (pahse == 'train' or phase == 'val' or pahse == 'test')
        self.phase = phase
        # transform
        self.imgs_trans = imgs_trans
        self.mask_trans = masks_trans
        
        # the path contains all data
        root = config.root
        if self.phase != 'test':
            imgs = []
            masks = []
            paths = glob.glob(root+'stage1_train/*')
            # val/train 3/10
            for p in paths:
                imgs.append(glob.glob(p+'/images/*.png')[0])
                masks = p + '/masks'
            if self.phase == 'train':
                self.imgs = imgs
                self.masks = masks
                self.imgs_num = len(self.imgs)
            else:
                self.imgs = imgs[int(0.7*len(imgs)):]
                self.masks = masks[int(0.7*len(masks)):]
                self.imgs_num = len(self.imgs)
        else:
            self.imgs = glob.glob(root + 'stage1_test/*/images/*.png')
            self.imgs_num = len(imgs)
    def __getitem__(self, index):
        '''
        data preprocess no resizing imgs
        use skimage.io read img,size is (height, width, 3)
        mask is gray image, so size is (height, width)
        for one image compose all masks to one
        return img(3, 128, 128) mask(128, 128) if not test else img(3, height, width)
        random crop to (3, 128, 128), filp, rotate
        '''
        img = io.imread(self.imgs[index])
        # just leave the 0 1 2 channel 
        if img.shape[2] > 3:
            assert(img[:, :, 3]!=255).sum()==0
        img = img[:, :, :3]
        if self.imgs_trans is None:
            self.imgs_trans = transforms.Compose([
                # torchvision's transforms
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if self.masks_trans is None:
            self.masks_trans = transforms.Compose([             
                transforms.ToTensor()
            ])
        # test phase has no mask
        if phase == 'test':
            
            # img name
            return self.imgs_trans(img), self.imgs[index].split('/')[-3]
        # compose mask and trans
        # copy from Tutorial on DSB2018
        else:
            # get this img's mask path object
            # list of all mask file
            mask_files = glob.glob(self.masks[index]+'*.png')
            masks = []
            for mask in mask_files:
                mask = io.imread(mask)
                # just verify mask contain 0 or 255
                # assert (mask[(mask!=0)]==255).all()
                masks.append(mask)
            
            #tmp_mask = mask.sum(0)
            for ii, mask in enumerate(masks):
                masks[ii] = mask / 255 * (ii+1)
            # mask ndarray
            # ndarray.sum(0) sum the first axis
            # that is this axis disappear other dimension's size doesn't change
            mask = masks.sum(0)
            return self.imgs_trans(img), self.masks_trans(mask)

    def __len__(self):
        return self.imgs_num

