import torch
import CNN
import clutchDataset
import CNN_metrics
import sys

class_mapping = CNN.class_mapping

net = CNN.CNN()
state_dict = torch.load('CNN_3ch_by2.pth')
net.load_state_dict(state_dict)

test_data = clutchDataset.test_dataset
train_data = clutchDataset.train_dataset
valid_data = clutchDataset.valid_dataset
##
y_true = []
y_pred = []

for i in range(len(test_data)):
    input, target = test_data[i][0], test_data[i][1]
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


print('Valid data: ')
y_true = []
y_pred = []

for i in range(len(valid_data)):
    input, target = valid_data[i][0], valid_data[i][1]
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
