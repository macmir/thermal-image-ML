import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import clutchDataset


data_path = 'data/clutch_2'

batch_size = 64
num_epochs = 100
learning_rate = .01

patience = 3

num_classes = 3
class_mapping = ['healthy', 'misalignment', 'rotor damage']
loss_val = []
val_loss_val = []

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# Preparing data

train_dataset = clutchDataset.train_dataset
test_dataset = clutchDataset.test_dataset
valid_dataset = clutchDataset.valid_dataset

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False)


# CNN class

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.BatchNorm2d(128),
            nn.Dropout(0.5),

            nn.Flatten(),

            nn.Linear(58240, 64),
            nn.ReLU(),
            nn.Linear(64, 3)

        )

    def forward(self, x):
        # for layer in self.net:
        #     x = layer(x)
        # print(x.size())
        x = self.net(x)
        return x


def train():
    last_loss = 100
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                loss_temp = []
                for val_input, val_label in valid_loader:
                    val_input = val_input.to(device)
                    val_label = val_label.to(device)
                    val_predictions = model(val_input)
                    val_loss = criterion(val_predictions, val_label)
                    loss_temp.append(val_loss.item())

        if loss.item() > last_loss:
            trigger += 1
        else:
            trigger = 0
        
        if trigger >= patience:
            print('Traning ended! - early stop')
            break

        last_loss = loss.item()
        val_loss_val.append(val_loss.item())
        loss_val.append(loss.item())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        print(f'Validation data loss: {val_loss.item()}')


if __name__ == '__main__':

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)

    train()

    torch.save(model.state_dict(), "saved_models/CNN_3ch_cropped_lr_early_stop.pth")

    epochs = []
    for i in range(num_epochs):
        epochs.append(i + 1)
    plt.plot(loss_val, '-g')
    plt.plot(val_loss_val, '-r')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()
