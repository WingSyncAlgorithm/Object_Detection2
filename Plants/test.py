import torch
from pathlib import Path
from torchvision import datasets, models, transforms
import numpy as np
PATH_train = ".\Plants\\archive\\train_3"
PATH_valid = "c:\\Users\\USER\\Documents\\plants_dataset\\archive\\val_3"
PATH_valid = Path(PATH_valid)
transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
batch_size = 70
# train_data = datasets.ImageFolder(PATH_train)
valid_data = datasets.ImageFolder(PATH_valid)
# im = valid_data[1][0]
# lim = np.array(list(im.getdata()))
# print(lim.shape)
print(valid_data.class_to_idx)
'''
train_loader = torch.utils.data.DataLoader(
    train_data)
valid_loader = torch.utils.data.DataLoader(
    valid_data)
print(type(train_loader))

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels
    return mean, std

mean_train, std_train = get_mean_std(train_loader)
mean_valid, std_valid = get_mean_std(valid_loader)

data_transforms_train_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_train, std=std_train)
])
data_transforms_valid_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_valid, std=std_valid)
])

train_data = datasets.ImageFolder(PATH_train,transform=data_transforms_train_norm)
valid_data = datasets.ImageFolder(PATH_valid,transform=data_transforms_valid_norm)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True)

print(train_data.class_to_idx)
print(type(train_loader))
'''