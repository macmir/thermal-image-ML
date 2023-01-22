import torch
import CNN
import clutchDataset
import CNN_metrics
import sys
import torchvision.models
import torch.nn as nn
import timm
import early_stopping
import matplotlib.pyplot as plt
import feedwor

import feedwor

class_mapping = CNN.class_mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_no = 14

if model_no == 1:
    net = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
                                                 in_channels=3, n_classes=3)

    # net = timm.create_model('efficientnet_cc_b1_8e', pretrained=True, num_classes=3)
elif model_no == 2:
    net = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    net.fc = net.fc.cuda()
elif model_no == 3:
    net = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    net.fc = net.fc.cuda()
elif model_no == 4:
    net = CNN.CNN()
elif model_no == 5:
    net = torchvision.models.convnext_tiny(weights = 'DEFAULT', in_channels = 3, out_channels = 3)
elif model_no == 6:
    net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, num_classes=1000)
    net.classifier[1] = nn.Linear(1280, 3)
elif model_no == 7:
    net = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.IMAGENET1K_V2, pretrained=True,
        num_classes=1000)
elif model_no == 8:
    net = timm.create_model('densenet201', pretrained=True, num_classes=3)

elif model_no == 9:
    net = torchvision.models.shufflenet_v2_x1_0(weights = 'DEFAULT', num_classes = 1000)
    net.fc = nn.Linear(1024, 3)

elif model_no == 10:
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    net.fc = net.fc.cuda()

elif model_no == 11:
    net = torchvision.models.densenet169(weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)
    net.classifier = nn.Linear(1664, 3)

elif model_no == 12:
    net = timm.create_model('resnet10t', pretrained=True, num_classes=3)

elif model_no ==13:
    net = CNN.CNN()

elif model_no == 14:
    net = feedwor.FeedForwardNet()



state_dict = torch.load('saved_models/FeedFrwrd_2.pth')

net.load_state_dict(state_dict)
net = net.to(device)

test_data = clutchDataset.test_dataset
train_data = clutchDataset.train_dataset
valid_data = clutchDataset.valid_dataset


print("Test data")
y_true = []
y_pred = []
y_correct = 0
y_correct_vector = []
for i in range(len(test_data)):
    input, target = test_data[i][0], test_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)

        predictions = net(input)

        predicted_index = predictions[0].argmax(0)

        if target == predicted_index:
            y_correct += 1

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)
    y_correct_vector.append(y_correct)
CNN_metrics.get_metrics(y_true, y_pred, class_mapping)
print('Accuracy: ', round( y_correct / len(test_data), 4) * 100, '%')
# print(len(y_correct_vector))
# accuracy = []
# for i in range(2):
#     accuracy[i] = y_correct_vector[i] / len(test_data) * 100

# plt.title("Test accuracy")
# plt.ylabel("Accuracy [%]")
# plt.xlabel("Epoch")
# plt.plot(accuracy * 100, '-bx')
# plt.show()
# print('Valid data: ')
# y_true = []
# y_pred = []
y_correct = 0

for i in range(len(valid_data)):
    input, target = valid_data[i][0], valid_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)
        predictions = net(input)
        predicted_index = predictions[0].argmax(0)

        if target == predicted_index:
            y_correct += 1

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)

CNN_metrics.get_metrics(y_true, y_pred, class_mapping)
print('Accuracy: ', round( y_correct / len(valid_data), 4) * 100, '%')

print('Training data:')
y_true = []
y_pred = []
y_correct = 0

for i in range(len(train_data)):
    input, target = train_data[i][0], train_data[i][1]
    input = input.to(device)
    target = target.to(device)
    net.eval()
    with torch.no_grad():
        input = torch.unsqueeze(input, 0)
        predictions = net(input)
        predicted_index = predictions[0].argmax(0)

        if target == predicted_index:
            y_correct += 1

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    y_true.append(expected)
    y_pred.append(predicted)

CNN_metrics.get_metrics(y_true, y_pred, class_mapping)
print('Accuracy: ', round( y_correct / len(train_data), 4) * 100, '%')
