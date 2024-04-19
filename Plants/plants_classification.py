import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import datasets,  transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)
PATH_train = "c:\\Users\\USER\\Documents\\plants_dataset\\archive\\train_3"
PATH_valid = "c:\\Users\\USER\\Documents\\plants_dataset\\archive\\val_3"
PATH_train = Path(PATH_train)
PATH_valid = Path(PATH_valid)
#TRAIN = Path(PATH_train)
# Batch：每批丟入多少張圖片
batch_size = 50
# Learning Rate：學習率
LR = 0.0001
#load data
transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
                                #  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
train_data = datasets.ImageFolder(PATH_train, transform=transforms)
valid_data = datasets.ImageFolder(PATH_valid, transform=transforms)
# print(train_data.class_to_idx)
# 切分70%當作訓練集、30%當作驗證集
# train_size = int(0.8 * len(train_data))
# valid_size = len(train_data) - train_size
# train_data, valid_data = torch.utils.data.random_split(
#     train_data, [train_size, valid_size])
# Dataloader可以用Batch的方式訓練
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True)
print(train_data.class_to_idx)
print(valid_data.class_to_idx)
class CNN_Model(nn.Module):
    # 列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 48, kernel_size=11, stride=1)
        self.relu1 = nn.Tanh()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(48, 8, kernel_size=8, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(8 * 26 * 26, 3) #input,output
    # 列出forward的路徑，將init列出的層代入

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
train_on_gpu = torch.cuda.is_available()
print("train_on_gpu",train_on_gpu)

def train(model, n_epochs, train_loader, valid_loader, optimizer, criterion):
    train_acc_his, valid_acc_his = [], []
    train_losses_his, valid_losses_his = [], []
    for epoch in range(1, n_epochs + 1):
        train_loss, valid_loss = 0.0, 0.0
        train_losses, valid_losses = [], []
        train_correct, val_correct, train_total, val_total = 0, 0, 0, 0
        train_pred, train_target = torch.zeros(8, 1), torch.zeros(8, 1)
        val_pred, val_target = torch.zeros(8, 1), torch.zeros(8, 1)
        count = 0
        count2 = 0
        print('running epoch: {}'.format(epoch))

        model.train()
        for data, target in tqdm(train_loader): #進度條
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            pred = output.data.max(dim=1, keepdim=True)[1]
            train_correct += np.sum(np.squeeze(
                pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * data.size(0))
            optimizer.zero_grad()
            if count == 0:
                train_pred = pred
                train_target = target.data.view_as(pred)
                count = count + 1
            else:
                train_pred = torch.cat((train_pred, pred), 0)
                train_target = torch.cat(
                    (train_target, target.data.view_as(pred)), 0)
        train_pred = train_pred.cpu().view(-1).numpy().tolist()
        train_target = train_target.cpu().view(-1).numpy().tolist()

        model.eval()
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
                val_pred = torch.cat((val_pred, pred), 0)
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

model1 = CNN_Model().to(device)
n_epochs = 10
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model1 = train(
    model1, n_epochs, train_loader, valid_loader, optimizer1, criterion)


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

model1.eval()  # 設定模型為評估模式
all_predictions = []  # 用於儲存所有預測結果
all_targets = []  # 用於儲存所有真實標籤

with torch.no_grad():
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
if_save = input('Do you want to save model press yes/no :\n')
if if_save == 'yes' or if_save == 'y':
    torch.save(model1, "full_trained_model.pth")

