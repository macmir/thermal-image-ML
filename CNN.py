import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import clutchDataset


data_path = 'data/clutch_2'
batch_size = 64
num_classes = 3
learning_rate = .001
num_epochs = 20
loss_val = []

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

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
        # for layer in self.net:
        #     x = layer(x)
            #print(x.size())
        x = self.net(x)
        return x


# Setting hyperparams

model = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

total_step = len(train_loader)

# Training

# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     loss_val.append(loss.item())
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

net = CNN(num_classes).to(device)
state_dict = torch.load("CNN.pth")
net.load_state_dict(state_dict)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train images: {} %'.format(100 * correct / total))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the valid images: {} %'.format(100 * correct / total))
    #torch.save(model.state_dict(), "CNN.pth")
    # epochs = []
    # for i in range(num_epochs):
    #     epochs.append(i+1)
    # plt.plot(epochs, loss_val)
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.show()
