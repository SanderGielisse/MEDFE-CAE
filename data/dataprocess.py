import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import os

class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, mask_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize(opt.fineSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize(opt.fineSize),
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt

        if train:
            self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
            self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
            self.mask_paths = sorted(glob('{:s}/*'.format(mask_root), recursive=True))
            self.Train=True

            """
            # only keep first 90% rest is validation
            de_split = int(len(self.de_paths) * 0.9)
            self.de_paths = self.de_paths[0:de_split]

            st_split = int(len(self.st_paths) * 0.9)
            self.st_paths = self.st_paths[0:st_split]

            mask_split = int(len(self.mask_paths) * 0.9)
            self.mask_paths = self.mask_paths[0:mask_split]
            """

        self.N_mask = len(self.mask_paths)
        print('N_de ', len(self.de_paths))
        print('N_st ', len(self.st_paths))
        print('N_mask ', len(self.mask_paths))

        self.st_map = {}
        for path in self.st_paths:
            _, tail = os.path.split(path)
            self.st_map[tail] = path

        self.mask_img = torch.empty([3, 256, 256], dtype=torch.float32) # self.mask_transform(mask_img.convert('RGB'))
        # print('shape ', mask_img.shape)
        self.mask_img[:, :, :] = 0.0
        self.mask_img[:, 64:(128+64), 64:(128+64)] = 1.0

    def __getitem__(self, index):

        de_path = self.de_paths[index]
        _, tail_de = os.path.split(de_path)

        st_path = self.st_map[tail_de]
        _, tail_st = os.path.split(st_path)

        if tail_de != tail_st:
            raise Exception("Wrongly matched file name?", de_path, tail_de, st_path, tail_st)

        de_img = Image.open(de_path)
        st_img = Image.open(st_path)
        # mask_img = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        de_img = self.img_transform(de_img.convert('RGB'))
        st_img = self.img_transform(st_img .convert('RGB'))
        
        #print('min ', torch.amin(de_img), torch.amin(st_img))
        #print('max ', torch.amax(de_img), torch.amax(st_img))


        return de_img, st_img, self.mask_img

    def __len__(self):
        return len(self.de_paths)
