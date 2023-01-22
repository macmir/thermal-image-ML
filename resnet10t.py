import timm
import sys
import torch
import torchvision.models
import torch.nn as nn
import clutchDataset
import matplotlib.pyplot as plt
import early_stopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('resnet10t', pretrained=True, num_classes=3)
model = model.to(device)

epochs = 150
learning_rate = 0.0003
batch_size = 8

loss_val = []
val_loss_val = []

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

early_stopper = early_stopping.EarlyStopper(patience=8, min_delta=0.1)

def train():

    for epoch in range(epochs):
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if labels == outputs[0].argmax(0):
                correct += 1

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
        test = val_loss / len(test_loader)
        if early_stopper.early_stop(val_loss/len(test_loader)):
            print('Training stopped - early stop!')
            break


if __name__ == '__main__':
    train()
    torch.save(model.state_dict(), "saved_models/resnet10t_20ep_for_resultspth")

    plt.plot(loss_val, '-g', label = 'training loss')
    plt.plot(val_loss_val, '-r', label = 'validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


