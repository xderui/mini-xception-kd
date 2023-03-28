from cProfile import label
import os
from statistics import mode
from turtle import forward
import torch
import datetime
import numpy as np
from visualdl import LogWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils.Model import mini_XCEPTION
from utils.dataset import FER2013
import torch.nn as nn

num_epochs = 200
log_step = 100      # 打印info的间隔步数
num_workers = 16    # 线程数

# output文件夹，会根据当前时间命名文件夹。
base_path = 'output/{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
writter = LogWriter(logdir=base_path)

batch_size = 64
input_size = (48, 48)
num_classes = 10
patience = 50

if not os.path.exists(base_path):
    os.makedirs(base_path)

# 定义模型
from resnet import ResNet101
from SeResNet import seresnet101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mini_XCEPTION(num_classes=10)
model.load_state_dict(torch.load('weights/KD_minixception2.pth', map_location=device))
# teacher_model = ResNet101(num_classes=10)
teacher_model = seresnet101()
teacher_model.load_state_dict(torch.load('weights/seresnet_final3.pth', map_location=device))
model.to(device)
teacher_model.to(device)

# 数据加载
train_dataset = FER2013("train", input_size=input_size)
test_dataset = FER2013("test", input_size=input_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 优化器
optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
# loss_fn = torch.nn.CrossEntropyLoss()

class categorical_crossentropy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,pred,label):
         return torch.sum(-1*label * torch.log(pred))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        # P = inputs


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        alpha = self.alpha

        probs = (P*targets).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class MSELOSS2(nn.Module):

    def __init__(self):
        super(MSELOSS2, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, inputs, targets,mask):
        loss = self.mseloss(inputs,targets)*mask
        print(loss)
        print(loss.sum()/(len(mask)*10))




# loss_fn = categorical_crossentropy()
# alpha
import pandas as pd
data_df = pd.read_csv('dataset/label.csv')
weights = list(data_df.iloc[:,2:].sum(axis=0))
weights_sum = sum(weights)
alpha = torch.Tensor([(weights_sum-i)/(9*weights_sum) for i in weights])
loss_fn = FocalLoss(class_num=10,alpha=alpha)
# loss_guide = nn.MSELoss('mean')
# loss_guide = MSELOSS2()
loss_guide = nn.KLDivLoss(reduction='batchmean')

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=int(patience / 4),
                                                       verbose=True)


def train_f():
    # 训练
    best_acc = 0
    step = 0
    for Epoch in range(0, num_epochs):
        total_train_loss, total_test_loss = 0, 0
        total_train_acc, total_test_acc = 0, 0
        count = 0
        end_index = len(train_loader) - 1
        model.train()
        for index, (labels, imgs) in enumerate(train_loader):
            imgs = imgs.to(device)
            T = 2
            labels_pd = model(imgs)
            labels_teacher = teacher_model(imgs)

            # print(labels_pd)
            # 记录acc和loss
            pred_idx = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)
            label_idx = np.argmax(labels,axis=-1)
            acc = accuracy_score(pred_idx, label_idx)
            # mask = 1 - (pred_idx == label_idx.cpu().detach().numpy()).astype(np.int)
            # mask =  torch.Tensor(mask).to(device)
            # print(pred_idx)
            # print(label_idx.cpu().detach().numpy())
            # print(acc)
            total_train_acc += acc
            loss1 = loss_fn(labels_pd, labels.to(device))
            # loss2 = loss_guide(labels_pd.float(),labels_teacher.float())
            labels_pd = F.log_softmax(labels_pd/T,dim=1)
            labels_teacher = F.softmax(labels_teacher/T,dim=1)
            loss2 = loss_guide(labels_pd.float(),labels_teacher.float())*T*T
            # print(loss2)
            loss = loss1*0.1 + loss2*0.9
            total_train_loss += loss.item()
            count += 1
            # 更新梯度
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_mean_acc = total_train_acc / count
            epoch_mean_loss = total_train_loss / count

            step += 1
            writter.add_scalar(tag="train_acc", step=step, value=epoch_mean_acc)
            writter.add_scalar(tag="train_loss", step=step, value=epoch_mean_loss)

            if index % log_step == 0 or index == end_index:
                print("e:{}\titer:{}/{}\tloss:{:.4f}\tacc:{:.4f}".format(Epoch, index, end_index,
                                                                         epoch_mean_loss,
                                                                         epoch_mean_acc))
        count = 0
        model.eval()
        for index, (labels, imgs) in enumerate(test_loader):
            labels_pd = model(imgs.to(device))
            acc = accuracy_score(np.argmax(labels_pd.cpu().detach().numpy(), axis=-1), np.argmax(labels,axis=-1))
            loss = loss_fn(labels_pd, labels.to(device))
            total_test_loss += loss.item()
            total_test_acc += acc
            count += 1

        mean_test_loss = total_test_loss / count
        mean_test_acc = total_test_acc / count
        print("evla\tloss:{:.4f}\tacc:{:.4f}".format(mean_test_loss, mean_test_acc))

        writter.add_scalar(tag="test_acc", step=Epoch, value=mean_test_acc)
        writter.add_scalar(tag="test_loss", step=Epoch, value=mean_test_loss)

        if (total_test_acc / count) > best_acc:
            torch.save(model.state_dict(), "{}/E{}_acc_{:.4f}.pth".format(base_path, Epoch, total_test_acc / count))
            best_acc = total_test_acc / count
            print("saved best model")


if __name__ == "__main__":
    train_f()
