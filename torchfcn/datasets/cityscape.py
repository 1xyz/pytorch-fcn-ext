import collections
import os.path as osp
import sys

import PIL.Image
import numpy as np
import torch
from torch.utils import data


class CityScape(data.Dataset):
    class_names = np.array([
        'ego vehicle',
        'rectification border',
        'out of roi',
        'static',
        'dynamic',
        'ground',
        'road',
        'sidewalk',
        'parking',
        'rail track',
        'building',
        'wall',
        'fence',
        'guard rail',
        'bridge',
        'tunnel',
        'pole',
        'polegroup',
        'traffic light',
        'traffic sign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'caravan',
        'trailer',
        'train',
        'motorcycle',
        'bicycle',
        'license plate'
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'CityScapes/CityScapes')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'CityScapes_%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                city = did.partition('_')[0]
                img_file = osp.join(
                    dataset_dir,
                    "{0}/{1}/{2}/{3}_{0}.png".format("leftImg8bit", split, city, did))
                lbl_file = osp.join(
                    dataset_dir,
                    "{0}/{1}/{2}/{3}_{0}_labelIds.png".format("gtFine", split, city, did))
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        with PIL.Image.open(img_file) as img:
            try:
                img = np.array(img, dtype=np.uint8)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                t = type(img)
                print(f"Current index {index} {img_file} type(image)={t}")
                raise
        # load label
        lbl_file = data_file['lbl']
        with PIL.Image.open(lbl_file) as lbl:
            try:
                lbl = np.array(lbl, dtype=np.int32)
                lbl[lbl == 255] = -1
            except:
                print("Unexpected error:", sys.exc_info()[0])
                t = type(img)
                print(f"Current index {index} {img_file} type(lbl)={t}")
                raise
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
