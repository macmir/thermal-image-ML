import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import clutchDataset
from sklearn import metrics
from torchmetrics import ConfusionMatrix
import torch.nn.functional as F

data_path = 'data/clutch_2'
batch_size = 64
num_classes = 3
learning_rate = .001
num_epochs = 20
loss_val = []

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

class_mapping = [
    "healthy",
    "misalignment",
    "rotor damage"
]

# Preparing data

train_dataset = clutchDataset.train_dataset
test_dataset = clutchDataset.test_dataset
valid_dataset = clutchDataset.valid_dataset

# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=True)


# CNN class
class CNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = self.net(x)
        print(x.size())
        for layer in self.net:
            x = layer(x)
            print(x.size())
        return x


# Setting hyperparams

# model = CNN(num_classes).to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# total_step = len(train_loader)

# Loading back saved model
model = CNN(num_classes).to(device)
state_dict = torch.load("CNN.pth")
model.load_state_dict(state_dict)

# Training

# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         # forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     loss_val.append(loss.item())
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Predicions
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        print(images.size())

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(class_mapping[predicted[predicted.argmax(0)].item()], class_mapping[labels[labels.argmax(0)].item()])
        y_pred.append(class_mapping[predicted[predicted.argmax(0)].item()])
        y_true.append(class_mapping[labels[labels.argmax(0)].item()])

    print('Accuracy of the network on the train images: {} %'.format(100 * correct / total))
    cm = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true, labels=["healthy", "misalignment", "rotor damage"])

    print(cm)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     y_pred = []
#     y_true = []
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


#     >>> target = torch.tensor([2, 1, 0, 0])
# >>> preds = torch.tensor([2, 1, 0, 1])
# >>> confmat = ConfusionMatrix(num_classes=3)
# >>> confmat(preds, target)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in valid_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the valid images: {} %'.format(100 * correct / total))

# Saving model
# torch.save(model.state_dict(), "CNN.pth")

# epochs = []
# for i in range(num_epochs):
#     epochs.append(i+1)
# plt.plot(epochs, loss_val)
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.show()
