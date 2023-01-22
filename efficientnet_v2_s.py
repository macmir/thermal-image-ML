import numpy as np
import timm
import torch
import torch.nn as nn
import clutchDataset
import matplotlib.pyplot as plt
import torchvision.models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT, in_channels = 3, n_classes = 3)
model = model.to(device)

epochs = 20
learning_rate = 0.0003
batch_size = 16

loss_val = []
val_loss_val = []
correct_predictions_vector = []

class_mapping = ['healthy', 'misalignment', 'rotor damage']

criterion = nn.CrossEntropyLoss()
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

    for epoch in range(epochs):
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

        correct_predictions_vector.append(correct_predictions)
        val_loss_val.append(val_loss.item())
        loss_val.append(loss.item())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        print(f'Validation data loss: {val_loss.item()}')


if __name__ == '__main__':
    train()
    torch.save(model.state_dict(), "saved_models/efficientnet_v2_s.pth")
    plt.title("Test and validation loss")
    plt.plot(loss_val, '-gx', label = 'training loss')
    plt.plot(val_loss_val, '-rx', label = 'validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.show()
    accuracy = np.zeros(epochs)

    for i in range(epochs):
        accuracy[i] = correct_predictions_vector[i] / len(test_dataset) * 100

    plt.title("Test accuracy")
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Epoch")
    plt.plot(accuracy, '-bx')
    plt.show()


