import torch
import CNN
import clutchDataset
from sklearn import metrics
from tabulate import tabulate


class_mapping = CNN.class_mapping

net = CNN.CNN()
state_dict = torch.load('CNN.pth')
net.load_state_dict(state_dict)

test_data = clutchDataset.test_dataset

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

cm = metrics.confusion_matrix(y_true, y_pred, labels=class_mapping)

total_predictions = [0, 0, 0]
total_true = [0, 0, 0]
for i in range(3):
    for j in range(3):
        total_predictions[i] += cm[j][i]
        total_true[i] += cm[i][j]
data = [["", class_mapping[0], class_mapping[1], class_mapping[2], "Total actual"],
        [class_mapping[0], cm[0][0], cm[0][1], cm[0][2], total_true[0]],
        [class_mapping[1], cm[1][0], cm[1][1], cm[1][2], total_true[1]],
        [class_mapping[2], cm[2][0], cm[2][1], cm[2][2], total_true[2]],
        ["Total predicted:", total_predictions[0], total_predictions[1], total_predictions[2], sum(total_true)]]
print(tabulate(data, tablefmt="simple_grid"))
print(f"\nRecall for healthy class: {cm[0][0]/total_predictions[0]}")
print(f"Recall for misalignment class: {cm[1][1]/total_predictions[1]}")
print(f"Recall for rotor damage class: {cm[2][2]/total_predictions[2]}")
print(f"\nPrecision for healthy class: {cm[0][0]/total_true[0]}")
print(f"Precision for misalignment class: {cm[1][1]/total_true[1]}")
print(f"Precision for rotor damage class: {cm[2][2]/total_true[2]}")




