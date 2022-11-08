import torch
import torch.nn as nn
import clutchDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = .001

data_path = 'data/clutch_2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_val = []
class_mapping = ['healthy', 'misalignment', 'rotor damage']

train_dataset = clutchDataset.train_dataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(142464, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, input):
        predictions = self.net(input)
        return predictions


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size, True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        predictions = model(input)
        loss = loss_fn(predictions, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f'Loss: {loss.item()}')
    loss_val.append(loss.item())


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print()
    print('Training done!')


if __name__ == '__main__':
    train_data_loader = create_data_loader(train_dataset, BATCH_SIZE)

    net = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9)

    train(net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(net.state_dict(), 'CNN.pth')
    print('Model trained and stored at CNN.pth')
