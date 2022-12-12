import torch
import torchvision.models
import torch.nn as nn
import clutchDataset
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT, in_channels = 3, n_classes = 3)
model = model.to(device)

epochs = 30
learning_rate = 0.001
batch_size = 64

loss_val = []
val_loss_val = []

class_mapping = ['healthy', 'misalignment', 'rotor damage']

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

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

def train():
    last_loss = 100

    for epoch in range(epochs):
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
        last_loss = loss.item()
        val_loss_val.append(val_loss.item())
        loss_val.append(loss.item())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        print(f'Validation data loss: {val_loss.item()}')

if __name__ == '__main__':
    
    train()
    torch.save(model.state_dict(), "saved_models/efficientnet_v2_s_1.pth")

    epochs = []
    for i in range(epochs):
        epochs.append(i + 1)
    plt.plot(loss_val, '-g')
    plt.plot(val_loss_val, '-r')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


