import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from torchvision.transforms import transforms
from tabulate import tabulate

# 0-healthy, 1-misalignment, 2-rotor
data_path = 'data/clutch_2'

# reading annotations file
af = pd.read_csv('annotations_file.csv')

# splitting data into 3 sets
train_data = af[af["dataset"].str.contains('train')]
test_data = af[af["dataset"].str.contains('test')]
val_data = af[af["dataset"].str.contains('val')]


class clutchDataset(Dataset):

    def __init__(self, dataframe, img_dir, is_train, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx, 1])

        img = cv2.imread(img_path)
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        image = Image.fromarray(img_norm)
        # label = self.dataframe.iloc[idx, 2]
        label = torch.tensor(int(self.dataframe.iloc[idx, 2]))
        # if self.is_train:
        #     label = torch.tensor(int(self.dataframe.iloc[idx, 2]))
        # else:
        #     label = torch.tensor(1)

        if self.transform:
            image = self.transform(image)

        return image, label


transform_train = transforms.Compose([
    transforms.Resize((225, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomRotation((1, 10)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_valid = transforms.Compose([
    transforms.Resize((225, 180)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((225, 180)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = clutchDataset(train_data, data_path, True, transform_train)
valid_dataset = clutchDataset(val_data, data_path, False, transform_valid)
test_dataset = clutchDataset(test_data, data_path, False, transform_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
if train_labels[0] == 0:
    label = 'healthy'

if train_labels[0] == 1:
    label = 'misalignment'

if train_labels[0] == 2:
    label = 'rotor damage'

# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()

# print(f"Label: {label}")
# print(train_dataset)
