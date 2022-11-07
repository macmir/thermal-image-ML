import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import clutchDataset
import matplotlib.pyplot as plt


BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = .001

data_path = 'data/clutch_2'
loss_val = []
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(225*180, 512),
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


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):  # training epoch of model
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        predictions = model(input)
        loss = loss_fn(predictions, target)  # predictions and expected targets - calculating the loss

        # backpropagate loss and update weights (gradient descent)
        optimiser.zero_grad()  # at every iteration we calculate gradient, zero_grad() sets gradients to zero
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")
    loss_val.append(loss.item())

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-" * 20)
    print("Training is done!")

if __name__ == "__main__":
    train_data = clutchDataset.train_dataset
    train_data_loader = create_data_loader(train_data, BATCH_SIZE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net = FeedForwardNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")
    epochs = []
    for i in range(EPOCHS):
        epochs.append(i+1)
    plt.plot(epochs, loss_val)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()