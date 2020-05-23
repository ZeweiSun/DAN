from torch.utils.data import Dataset, DataLoader
import time
import math
import numpy as np
import os
import torch
import cv2


class _360CC(Dataset):
    def __init__(self, img_list, img_height, img_width, augment = False, is_train=True):

        self.root = img_list
        self.is_train = is_train
        self.inp_h = img_height
        self.inp_w = img_width

        self.dataset_name = '360_CC'

        self.mean = np.array(0.588, dtype=np.float32)
        self.std = np.array(0.193, dtype=np.float32)

        txt_file =self.root+'/train.txt'  if is_train else self.root+'/test.txt'

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([idx[:-1] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img_label = list(self.labels[idx].values())[0]
        img = cv2.imread(os.path.join(self.root+'/images', img_name),0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        sample = {'image': torch.from_numpy(img), 'label': img_label}

        return sample
