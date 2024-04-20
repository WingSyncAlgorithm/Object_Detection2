import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
import pandas as pd
import os
from os import listdir
from tqdm import tqdm
from PIL import Image


# In[8]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#这个device的用处是作为Tensor或者Model被分配到的位置(判斷店電腦是否有gpu，true的話就是有)
PATH_train = R"archive\training_set\training_set"
TRAIN = Path(PATH_train)
# Batch：每批丟入多少張圖片
batch_size = 8
# Learning Rate：學習率
LR = 0.0001
transforms = transforms.Compose(#把兩個變換組合在一起定義為transforms
    [transforms.Resize((224, 224)), transforms.ToTensor()])#切割(統一大小)並將影像轉為Tenso


# In[9]:


train_data = datasets.ImageFolder(PATH_train, transform=transforms)#從PATH_train路徑尋找圖片，對圖片做transforms的轉換操作
# print(train_data.class_to_idx)
# 切分70%當作訓練集、30%當作驗證集
train_size = int(0.7 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(#將train_data裡面的數據分成7成3成
    train_data, [train_size, valid_size])
# Dataloader可以用Batch的方式訓練
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)#從train_data裡面，每次丟出batch_size個資料，shuffle=True意思是使資料在每次重新洗牌
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True)


# In[10]:


class CNN_Model(nn.Module):
    # 列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)#卷積層，輸入三個通道(rgb)，輸出為16個特徵圖，卷積核為5*5
        self.relu1 = nn.ReLU(inplace=True)#線性整流層(像素為正的地方不動，為負的地方變為0)
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)#池化層(最大池化)
        # Convolution 2
        self.cnn2 = nn.Conv2d(16, 8, kernel_size=11, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(8 * 50 * 50, 2)#全連接層
    # 列出forward的路徑，將init列出的層代入

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)#做全連接層前，得先將圖攤成一維
        out = self.fc(out)
        return out


# In[11]:


train_on_gpu = torch.cuda.is_available()#確認你的 GPU，是否支持 CUDA


def train(model, n_epochs, train_loader, valid_loader, optimizer, criterion):#model:神經網路，n-epochs:我們要訓練幾個epoch，criterion:代價函數
    train_acc_his, valid_acc_his = [], []
    train_losses_his, valid_losses_his = [], []
    for epoch in range(1, n_epochs + 1):#定義訓練幾個epoch，太多會過擬合
        train_loss, valid_loss = 0.0, 0.0#初始化
        train_losses, valid_losses = [], []
        train_correct, val_correct, train_total, val_total = 0, 0, 0, 0
        train_pred, train_target = torch.zeros(8, 1), torch.zeros(8, 1)#8*1的0矩陣
        val_pred, val_target = torch.zeros(8, 1), torch.zeros(8, 1)
        count = 0
        count2 = 0
        print('running epoch: {}'.format(epoch))

        model.train()#設定為訓練模式(會反向傳播)
        for data, target in tqdm(train_loader):#data:每張圖片，target:代表正確答案
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)#output:前向傳播預測值
            loss = criterion(output, target)#損失函數
            pred = output.data.max(dim=1, keepdim=True)[1]#較高的值代表預測結果
            train_correct += np.sum(np.squeeze(#存下準確度
                pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            loss.backward()#反向傳播
            optimizer.step()#更新權重(可以設為梯度下降法)
            train_losses.append(loss.item() * data.size(0))
            optimizer.zero_grad()#梯度規0
            if count == 0:
                train_pred = pred
                train_target = target.data.view_as(pred)
                count = count + 1
            else:
                train_pred = torch.cat((train_pred, pred), 0)#張亮拼接(豎著)
                train_target = torch.cat(
                    (train_target, target.data.view_as(pred)), 0)
        train_pred = train_pred.cpu().view(-1).numpy().tolist()#攤平張亮，並轉換為列表
        train_target = train_target.cpu().view(-1).numpy().tolist()

        model.eval()#設為評估模式，會少掉loss.backward()和optimizer.step()這兩步
        for data, target in tqdm(valid_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            pred = output.data.max(dim=1, keepdim=True)[1]
            val_correct += np.sum(np.squeeze(
                pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item() * data.size(0))
            if count2 == 0:
                val_pred = pred
                val_target = target.data.view_as(pred)
                count2 = count + 1
            else:
                val_pred = torch.cat((val_pred, pred), 0)#張亮拼接(豎著)
                val_target = torch.cat(
                    (val_target, target.data.view_as(pred)), 0)
        # 轉list
        val_pred = val_pred.cpu().view(-1).numpy().tolist()
        val_target = val_target.cpu().view(-1).numpy().tolist()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        train_acc = train_correct / train_total
        valid_acc = val_correct / val_total
        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)

        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            train_acc, valid_acc))

    return train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model


# In[12]:


# model1 = CNN_Model()  # 不需要指定 .cpu()，因為預設就在 CPU 上
# 在模型初始化時將模型移動到相應的設備（GPU 或 CPU）
model1 = CNN_Model().to(device)#將CNN_Model丟到gpu或cpu上
n_epochs = 30
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)#使用Adam法更新權重
criterion = nn.CrossEntropyLoss()#交叉商損失函數

train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model1 = train(
    model1, n_epochs, train_loader, valid_loader, optimizer1, criterion)


# In[22]:


plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.plot(train_losses_his, 'b', label='training loss')
plt.plot(valid_losses_his, 'r', label='validation loss')
plt.title("Simple CNN Loss")
plt.legend(loc='upper left')
plt.subplot(222)
plt.plot(train_acc_his, 'b', label='training accuracy')
plt.plot(valid_acc_his, 'r', label='validation accuracy')
plt.title("Simple CNN Accuracy")
plt.legend(loc='upper left')
plt.show()
#畫出學習曲線

# In[31]:


model1.eval()  # 設定模型為評估模式
all_predictions = []  # 用於儲存所有預測結果
all_targets = []  # 用於儲存所有真實標籤

with torch.no_grad():#不計算梯度，並執行以下程式
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model1(data)
        predictions = output.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
conf_matrix = confusion_matrix(all_targets, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()


# In[21]:


# 在訓練完畢後，保存完整模型（包括結構和參數）到指定文件
torch.save(model1, 'full_trained_model.pth')
