'''
author: meng-zha
data: 2020/06/01
'''

import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ED

class FaceMaskDataset(Dataset):
    def __init__(self, root_path,list_path,transform):
        self.root_path = root_path
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

    def __len__(self):
        return len(self.img_ids)

    def load_images(self,idx):
        image_name = self.img_ids[idx]+'.jpg'
        image_path = os.path.join(self.root_path,image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_labels(self,idx):
        annotations = np.zeros((0, 5))

        label_name = self.img_ids[idx]+'.xml'
        label_path = os.path.join(self.root_path,label_name)
        tree = ED.ElementTree(file=label_path)
        objects = tree.findall('object')

        for obj in objects:
            bndbox = obj.find('bndbox')
            annotation = np.zeros((1,5))
            annotation[0,0] = int(bndbox.find('xmin').text)
            annotation[0,1] = int(bndbox.find('ymin').text)
            annotation[0,2] = int(bndbox.find('xmax').text)
            annotation[0,3] = int(bndbox.find('ymax').text)
            annotation[0,4] = self.class2num(str(obj.find('name').text))

            annotations = np.append(annotations,annotation,axis=0)

        return annotations

    def class2num(self,cl):
        c2n = {'face_mask':0,'face':1}
        return c2n[cl]

    def __getitem__(self,idx):
        img = self.load_images(idx)
        anno = self.load_labels(idx)
        sample = {'img': img, 'annot': anno}
        if self.transform:
            sample = self.transform(sample)
        return sample