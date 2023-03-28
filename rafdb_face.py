import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from utils.DataAugment import *
from torchvision.transforms import *

emotion_dict = {'surprise':0,'fear':1,'disgust':2,'happy':3,'sad':4,'angry':5,'neutral':6}

class RAFDBFACE(data.Dataset):
    def __init__(self,mode='train'):
        self.img = []
        self.target = []
        root_path = './dataset/RAFDB/basic/'
        txt_path = root_path + 'list_patition_label.txt'
        img_path = root_path + 'aligned/'
        f = open(txt_path,'r')
        lines = f.readlines()
        for line in lines:
            img_label = line.split(' ')
            if img_label[0].split('_')[0] != mode:
                continue
            self.img.append(img_path + img_label[0].split('.')[0] + '_aligned.jpg')
            self.target.append(int(img_label[1])-1)

        self.aug = Augment([Salt_Pepper_Noise(0.05),
                                Width_Shift_Range(0.1),
                                Height_Shift_Range(0.1)])

        self.transform = transforms.Compose([ToTensor(),
                                                 ColorJitter(brightness=0.2),
                                                 RandomRotation(10),
                                                 RandomHorizontalFlip(0.5)])


    def __getitem__(self, index):
#        print(self.img)
        img = cv2.imread(self.img[index])
        # img = cv2.resize(img,(224,224,),interpolation=cv2.INTER_CUBIC)
       # img = img.transpose(2, 0, 1)
        #print(img.shape)
        img=self.aug(img)
        img=self.transform(img)
        img = np.float32(img)
        # label=np.zeros((1,7)).view(1,-1)
        label = [0 for i in range(7)]
        # print(label)
        # print(self.target[index])
        label[self.target[index]]=1
        return np.array(label),img


    def __len__(self):
        return len(self.target)


# dataset = RAFDBFACE()
# dataloader = iter(data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0))
# img,target = next(dataloader)
# print(img,target)
#


