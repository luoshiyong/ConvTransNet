import numpy as np
import cv2  #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import scipy.ndimage as ndi

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, val = False,transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.per = 1.0
        self.val = val
    def __len__(self):
        return int(self.per*len(self.img_paths))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        filename = (mask_path.split('\\')[-1]).split('.')[0]
        if not self.val:
            flip_num = np.random.randint(0, 8)
            if flip_num == 1:
                npimage = np.flipud(npimage)
                npmask = np.flipud(npmask)
            elif flip_num == 2:
                npimage = np.fliplr(npimage)
                npmask = np.fliplr(npmask)
            elif flip_num == 3:
                npimage = np.rot90(npimage, k=1, axes=(1, 0))
                npmask = np.rot90(npmask, k=1, axes=(1, 0))
            elif flip_num == 4:
                npimage = np.rot90(npimage, k=3, axes=(1, 0))
                npmask = np.rot90(npmask, k=3, axes=(1, 0))
            elif flip_num == 5:
                cropp_img = np.fliplr(npimage)
                cropp_tumor = np.fliplr(npmask)
                npimage = np.rot90(cropp_img, k=1, axes=(1, 0))
                npmask = np.rot90(cropp_tumor, k=1, axes=(1, 0))
            elif flip_num == 6:
                cropp_img = np.fliplr(npimage)
                cropp_tumor = np.fliplr(npmask)
                npimage = np.rot90(cropp_img, k=3, axes=(1, 0))
                npmask = np.rot90(cropp_tumor, k=3, axes=(1, 0))
            elif flip_num == 7:
                cropp_img = np.flipud(npimage)
                cropp_tumor = np.flipud(npmask)
                npimage = np.fliplr(cropp_img)
                npmask = np.fliplr(cropp_tumor)
        # npimage = npimage[32:480,32:480,:]
        # npmask = npmask[32:480,32:480]
        npimage = npimage.transpose((2, 0, 1))

        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        nplabel = np.empty((336,336,2))
        nplabel[:, :, 0] = liver_label
        nplabel[:, :, 1] = tumor_label
        nplabel = nplabel.transpose((2, 0, 1))


        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        if not self.val:
            return npimage,nplabel
        else:
            return npimage,nplabel,filename
