import torch
from torch import nn
from torch.utils.data import DataLoader
import clutchDataset
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = .001

data_path = 'data/clutch_2'
loss_val = []
loss_valid_vector = []
correct_predictions_vector = []
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(640 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, valid_loader, test_loader):  # training epoch of model
    correct_predictions = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        predictions = model(input)
        loss = loss_fn(predictions, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    #     with torch.no_grad():
    #         model.eval()
    #         for input_v, target_v in valid_loader:
    #             input_v, target_v = input_v.to(device), target_v.to(device)
    #
    #             predictions_v = model(input_v)
    #             loss_valid = loss_fn(predictions_v, target_v)
    #
    #         for input_t, target_t in test_loader:
    #             input_t, target_t = input_t.to(device), target_t.to(device)
    #
    #             predictions_t = model(input_t)
    #             _, pred_t = torch.max(predictions_t, dim=1)
    #             correct_predictions += torch.sum(pred_t == target_t).item()
    #         correct_predictions_vector.append(correct_predictions)
    #
    #
    # loss_valid_vector.append(loss_valid)
    print(f"Loss: {loss.item()}")
    loss_val.append(loss.item())


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, valid_data_loader, test_data_loader)
        print("-" * 20)
    print("Training is done!")


if __name__ == "__main__":
    train_data = clutchDataset.train_dataset
    valid_data = clutchDataset.valid_dataset
    test_data = clutchDataset.valid_dataset

    train_data_loader = create_data_loader(train_data, BATCH_SIZE)
    valid_data_loader = create_data_loader(valid_data, BATCH_SIZE)
    test_data_loader = create_data_loader(test_data, BATCH_SIZE)


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net = FeedForwardNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")

    plt.title("Test and validation loss")
    plt.plot(loss_val, '-gx', label = 'training loss')
    plt.plot(loss_valid_vector, '-rx', label = 'validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.show()
    accuracy = np.zeros(EPOCHS)

    for i in range(EPOCHS):
        accuracy[i] = correct_predictions_vector[i] / len(test_data) * 100

    plt.title("Test accuracy")
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Epoch")
    plt.plot(accuracy, '-bx')
    plt.show()
