import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import clutchDataset
import numpy as np

data_path = 'data/clutch_2'

batch_size = 32
num_epochs = 60
learning_rate = .0001

patience = 3

num_classes = 3
class_mapping = ['healthy', 'misalignment', 'rotor damage']
loss_val = []
val_loss_val = []
correct_predictions_vector = []
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

            nn.Linear(192000, 8),
            nn.ReLU(),
            nn.Linear(8, 3)

        )

    def forward(self, x):
        x = self.net(x)
        return x


def train():
    last_loss = 100
    for epoch in range(num_epochs):
        correct_predictions = 0
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

                for test_input, test_label in test_loader:
                    test_input = test_input.to(device)
                    test_label = test_label.to(device)
                    test_predictions = model(test_input)
                    _, pred_t = torch.max(test_predictions, dim=1)
                    correct_predictions += torch.sum(pred_t == test_label).item()
                correct_predictions_vector.append(correct_predictions)
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

    torch.save(model.state_dict(), "saved_models/CNN60.pth")

    plt.title("Test and validation loss")
    plt.plot(loss_val, '-gx', label='training loss')
    plt.plot(val_loss_val, '-rx', label='validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.show()
    accuracy = np.zeros(num_epochs)

    for i in range(num_epochs):
        accuracy[i] = correct_predictions_vector[i] / len(test_dataset) * 100

    plt.title("Test accuracy")
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Epoch")
    plt.plot(accuracy, '-bx')
    plt.show()
