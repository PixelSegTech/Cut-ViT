from src.utils.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms



def ADE20K(config):
    traindataset = FullDataset(
        config.dataset, "/data1/yjj/datasets/ADE20k/ADEChallengeData2016/images", "/data1/yjj/datasets/ADE20k/ADEChallengeData2016/annotations" , 'training', 512
    )

    valdataset = FullDataset(
        config.dataset, "/data1/yjj/datasets/ADE20k/ADEChallengeData2016/images", "/data1/yjj/datasets/ADE20k/ADEChallengeData2016/annotations" , 'validation', 512
    )
    return traindataset, valdataset


class FullDataset(Dataset):
    def __init__(self, root, mask, mode, size=None, id_path=None, nsample=None):
        self.root = root
        self.mask = mask
        self.mode = mode + "2017"
        self.size = size
    
        folder = "/data/COCO2017/train2017"  # 换成你的文件夹路径

        self.ids = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpg"):          # 只处理 .jpg
                name_no_ext = os.path.splitext(filename)[0]  # 去掉 .jpg
                self.ids.append(name_no_ext)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, self.mode, id + ".jpg")).convert('RGB')
        
        ignore_value =  255
        if self.mode == 'validation':
            img = crop(img, self.size, ignore_value)
            img = normalize(img)
            return img

        img = resize(img, (0.5, 2.0))
    
        img = crop(img, self.size, ignore_value)
        img = hflip(img, p=0.5)

        return normalize(img)

    def __len__(self):
        return len(self.ids)
