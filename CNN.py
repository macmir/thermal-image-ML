import sys

import torch
import torch.nn as nn
import clutchDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = .001

data_path = 'data/clutch_2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_val = []
loss_val_v = []
class_mapping = ['healthy', 'misalignment', 'rotor damage']

train_dataset = clutchDataset.train_dataset
valid_dataset = clutchDataset.valid_dataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, input):
        # for layer in self.net:
        #     input = layer(input)
        #     print(input.size())
        predictions = self.net(input)
        return predictions


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size, True)
    return train_dataloader


def train_single_epoch(model, data_loader, lossfn, opt, dev):
    for inp, target in data_loader:
        inp, target = inp.to(dev), target.to(dev)

        opt.zero_grad()
        predictions = model(inp)
        loss = lossfn(predictions, target)
        loss.backward()
        opt.step()
    print(f'Loss: {loss.item()}')
    loss_val.append(loss.item())


def train(model, data_loader, lossfn, optim, dev, epochs):
    for i in range(epochs):
        print(f'Epoch {i + 1}')
        train_single_epoch(model, data_loader, lossfn, optim, dev)

        print()
    print('Training done!')


if __name__ == '__main__':
    train_data_loader = create_data_loader(train_dataset, BATCH_SIZE)
    valid_data_loader = create_data_loader(valid_dataset, BATCH_SIZE)

    net = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9)

    train(net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(net.state_dict(), 'CNN_123.pth')
    print('Model trained and stored at CNN.pth')

    plt.plot(loss_val)
    plt.ylabel('loss')
    plt.show()
