import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from torchvision.transforms import transforms
from torchvision import transforms

from tabulate import tabulate


# 0-healthy, 1-misalignment, 2-rotor
data_path = 'data/clutch_2'
# af = pd.read_csv('annotations_file_n.csv')
# af = pd.read_csv('new_dataset.csv')
af = pd.read_csv('annotation_files/annotations_file.csv')
train_data = af[af["dataset"].str.contains('train')]
test_data = af[af["dataset"].str.contains('test')]
validation_data = af[af["dataset"].str.contains('val')]

# print(f'train data size: {len(train_data)}')
# print(f'test data size: {len(test_data)}')
# print(f'valid data size: {len(validation_data)}')
# print(f'total = {len(train_data) + len(test_data) + len(validation_data)}')

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
        image = transforms.functional.crop(image, 194, 372 , 140, 220) # my initial crop
        # image = transforms.functional.crop(image, 105, 328, 320, 320) # MP crop
        label = torch.tensor(int(self.dataframe.iloc[idx, 2]))

        if self.transform:
            image = self.transform(image)
        return image, label


width = int(640 / 1)
height = int(512 / 1)
# width = 320 #MP
# height = 320 #MP
# width = 220
# height = 140
transform_train = transforms.Compose([
    transforms.Resize((height, width)),
    # transforms.RandomAffine((1, 10)),
    # transforms.RandomGrayscale(0.1),
    # transforms.RandomVerticalFlip(0.2),
    # transforms.GaussianBlur((3, 3)),
    # transforms.RandomInvert(0.3),
    # transforms.RandomSolarize(100),
    # transforms.RandomAdjustSharpness(2),
    # transforms.RandomAutocontrast(0.2),
    # transforms.ColorJitter(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(),
    # transforms.RandomRotation((1, 10)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_valid = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


train_dataset = clutchDataset(train_data, data_path, True, transform_train)
valid_dataset = clutchDataset(validation_data, data_path, False, transform_valid)
test_dataset = clutchDataset(test_data, data_path, False, transform_test)
#
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#
# train_features, train_labels = next(iter(train_dataloader))
#
# img = train_features[0].squeeze()
# if train_labels[0] == 0:
#     label = 'healthy'
# if train_labels[0] == 1:
#     label = 'misalignment'
# if train_labels[0] == 2:
#     label = 'rotor damage'
#
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# print(f"Label: {label}")
# plt.show()
