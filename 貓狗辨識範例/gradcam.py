#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
import pandas as pd
import os
from os import listdir
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import matplotlib as mpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Model(nn.Module):
    #列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1) 
        self.relu1 = nn.ReLU(inplace=True) 
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(16,8, kernel_size=11, stride=1) 
        self.relu2 = nn.ReLU(inplace=True) 
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(8 * 50 * 50, 2)     
    #列出forward的路徑，將init列出的層代入
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
    
class GradCamModel(nn.Module):
    def __init__(self, custom_model):
        super(GradCamModel, self).__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # 使用你自己的模型，這裡假設 custom_model 是你的 CNN_Model 實例
        self.custom_model = custom_model

        # 註冊 layer4 的 forward hook
        self.layerhook.append(self.custom_model.cnn2.register_forward_hook(self.forward_hook()))

        # 將所有參數設置為需要梯度更新
        for p in self.custom_model.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            # 設定 selected_out，同時註冊 tensor hook
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.custom_model(x)
        return out, self.selected_out

# 載入你的自己訓練的模型，例如 'my_trained_model.pth'
loaded_model = torch.load('full_trained_model.pth')
loaded_model.eval()  # 設置模型為評估模式

# 讀取並預處理圖片
image_path = r"cat.jpg"
image = Image.open(image_path)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
input_image = transform(image).unsqueeze(0)  # 添加批次維度

# 將圖片輸入你的自己的模型進行預測
# 將圖片輸入你的自己的模型進行預測
with torch.no_grad():
    input_image = input_image.to(device)  # 將 input_image 移動到 GPU 上
    output = loaded_model(input_image)


# 獲得預測結果
predicted_class = torch.argmax(output, dim=1).item()

# 創建 GradCAM 模型實例，將你的自己的模型傳遞給它
gradcam_model = GradCamModel(loaded_model)

# 使用 GradCAM 模型計算 GradCAM 結果
out, acts = gradcam_model(input_image)
acts = acts.detach().cpu()

# 設定目標類別為預測類別
target_class = torch.tensor([predicted_class])
# 設定目標類別為預測類別
target_class = target_class.to(device)  # 將 target_class 移動到 GPU 上

# 計算梯度和 pooled gradients
loss = nn.CrossEntropyLoss()(out, target_class)
loss.backward()
grads = gradcam_model.get_act_grads().detach().cpu()
# 計算 pooled gradients
pooled_grads = torch.mean(grads, dim=0).detach().cpu()

for i in range(acts.shape[1]):
    acts[0, i] += pooled_grads[i]
# 計算 heatmap
heatmap_j = torch.mean(acts, dim=1).squeeze()
heatmap_j_max = heatmap_j.max()
heatmap_j /= heatmap_j_max
heatmap_j = heatmap_j.numpy()



# 獲取原圖尺寸
image_size = image.size

# 調整 heatmap_j 的尺寸，使其與原圖尺寸相同
heatmap_j = resize(heatmap_j, (image_size[1], image_size[0]), preserve_range=True)
cmap = mpl.cm.get_cmap('jet', 256)
heatmap_j2 = cmap(heatmap_j, alpha=0.7)

# 顯示原圖與 GradCAM 結果，調整 figsize 參數來改變圖片尺寸
fig, axs = plt.subplots(1, 1, figsize=(12, 12))
axs.imshow(image)
axs.imshow(heatmap_j2, cmap='jet', alpha=0.5)

# 顯示預測結果
class_names = ['cat', 'dog']
predicted_class_name = class_names[predicted_class]
plt.title(f'Predicted class: {predicted_class_name}', fontsize=24)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




