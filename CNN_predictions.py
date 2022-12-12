import torch
import CNN
import clutchDataset
import CNN_metrics
import sys
import torchvision.models
import torch.nn as nn


class_mapping = CNN.class_mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT, in_channels = 3, n_classes = 3)
net = torchvision.models.resnet34(pretrained = True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 3)
net.fc = net.fc.cuda()

state_dict = torch.load('saved_models/resnet34.pth')
net.load_state_dict(state_dict)
net = net.to(device)

test_data = clutchDataset.test_dataset
train_data = clutchDataset.train_dataset
valid_data = clutchDataset.valid_dataset
##
y_true = []
y_pred = []
h = []
for i in range(len(test_data)):
    input, target = test_data[i][0], test_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)

        
        predictions = net(input)
        
        predicted_index = predictions[0].argmax(0)
        if target == 0:
            h.append(predictions)

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)

CNN_metrics.get_metrics(y_true, y_pred, class_mapping)


print('Valid data: ')
y_true = []
y_pred = []

for i in range(len(valid_data)):
    input, target = valid_data[i][0], valid_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)
        predictions = net(input)
        predicted_index = predictions[0].argmax(0)

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)

CNN_metrics.get_metrics(y_true, y_pred, class_mapping)

print('Training data:')
y_true = []
y_pred = []

for i in range(len(train_data)):
    input, target = train_data[i][0], train_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)
        predictions = net(input)
        predicted_index = predictions[0].argmax(0)

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)

CNN_metrics.get_metrics(y_true, y_pred, class_mapping)
# for i in range(len(h)):
#     print(h[i]) 
