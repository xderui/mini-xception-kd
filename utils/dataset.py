# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/7 10:16   thgpddl      1.0         None
"""
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import *

from .DataAugment import *

hsr = Height_Shift_Range(0.1)
wsr = Width_Shift_Range(0.1)
from PIL import Image


class FER2013(Dataset):
    def __init__(self, mode, input_size):
        super(FER2013, self).__init__()
        self.mode = mode
        df = pd.read_csv(os.path.join("dataset", "new_data3.csv"))
        df_label = pd.read_csv(os.path.join("dataset", "new_label3.csv"))
        if mode == "train":
            self.data = np.array(df.loc[df['Usage']=='Training',['pixels']])
            self.label = df_label.loc[df_label['Usage']=='Training',df_label.columns[4:]].to_numpy()
        else:
            self.data = np.array(df.loc[df['Usage']=='PublicTest',['pixels']])
            self.label = df_label.loc[df_label['Usage']=='PublicTest',df_label.columns[4:]].to_numpy()
            # self.data = np.concatenate((np.array(df.loc[df['Usage']=='PublicTest',['pixels']]),np.array(df.loc[df['Usage']=='PrivateTest',['pixels']])),axis=0)
            # self.label = np.concatenate((df_label.loc[df_label['Usage']=='PublicTest',df_label.columns[3:]].to_numpy(),df_label.loc[df_label['Usage']=='PrivateTest',df_label.columns[3:]].to_numpy()),axis=0)


        # print(os.path.join("dataset", mode + ".csv"))
        self.input_size = input_size
        if mode == "train":
            self.aug = Augment([Salt_Pepper_Noise(0.05),
                                Width_Shift_Range(0.1),
                                Height_Shift_Range(0.1)])

            self.transform = transforms.Compose([ToTensor(),
                                                 ColorJitter(brightness=0.2),
                                                 RandomRotation(10),
                                                 RandomHorizontalFlip(0.5)])
        else:
            self.aug = Augment()
            self.transform = transforms.Compose([ToTensor()])

    def __getitem__(self, item):
        # label, img = self.data[item]
        # print(self.label[item])
        label = self.label[item]/10
        img = self.data[item][0]
        data = np.array([int(pix) for pix in img.split()], dtype=np.uint8)
        img = np.reshape(data, (48, 48))
        # img = Image.fromarray(img)
        # img.resize(self.input_size,Image.ANTIALIAS)
        # img = np.array(img)
        # np的resize是用0填充，所以64*64的下降的原因呢可能是这个
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)

        img = self.aug(img)  # 自定义增强
        img = self.transform(img)  # torch增强
        # img=(img-0.5)*2
        return label, img

    def __len__(self):
        return len(self.data)
