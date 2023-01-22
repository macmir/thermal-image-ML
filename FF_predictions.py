import sys
import torch
import FF
import clutchDataset
from sklearn import metrics
import numpy as np


class_mapping = [
    "healthy",
    "misalignment",
    "rotor damage"
]
good_prediction = 0
bad_prediction = 0
total = 0
healthy_guesses = 0
rotor_guesses = 0
misalignment_guesses = 0
healthy_tp = 0
rotor_tp = 0
misalignment_tp = 0
healthy_tot = 0
misalignment_tot = 0
rotor_tot = 0

def predict(model, input, target, class_mapping):
    model.eval()    # call it every time you want to make an inference
    with torch.no_grad():
        # print(input.size())

        predictions = model(input) # Tensor(1, 10) - 1-number of inputs, 10-number of classes
        # Tensor(1, 10) -> [[0.1, 0.1, 0.2 ... 0.6]] - sums up to 1.0 as we used softmax
        predicted_index = predictions[0].argmax(0) # take first max value
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        print(predicted, expected)
    return predicted, expected


if __name__ == "__main__":

    # load back the model
    feed_forward_net = FF.FeedForwardNet()
    state_dict = torch.load("saved_models/FeedFrwrd.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load validation dataset
    #validation_data = clutchDataset.valid_dataset
    validation_data = clutchDataset.test_dataset
    y_true = []
    y_pred = []
    # get a sample from the validation dataset for inference
    print(len(validation_data))
    for i in range(len(validation_data)):
        input, target = validation_data[i][0], validation_data[i][1]
    # make an inference
    #     print(input.size())

        predicted, expected = predict(feed_forward_net, input, target, class_mapping)
        # print(f"Predicted: '{predicted}', expected: '{expected}'")
        y_true.append(expected)
        y_pred.append(y_true)

    cm = metrics.confusion_matrix(y_true, y_pred, labels=["healthy", "misalignment", "rotor damage"])
    print("Confusion matrix:")
    print(cm)
    print(len(y_true))
    pred_sum = [0, 0, 0]
    actual_sum = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            pred_sum[j] += cm[i][j]
            actual_sum[j] += cm[j][i]
    print()
    print(pred_sum)
    print(actual_sum)
    print()
    print(f"Recall for 'healthy' class: {cm[0][0] / pred_sum[0]}")
    print(f"Precision for 'healthy class: {cm[0][0] / actual_sum[0]}")
    print(f"\nRecall for 'misalignment' class: {cm[1][1] / pred_sum[1]}")
    print(f"Precision for 'misalignment class: {cm[0][0] / actual_sum[1]}")
    print(f"\nRecall for 'rotor damage' class: {cm[2][2] / pred_sum[2]}")
    print(f"Precision for 'rotor damage class: {cm[0][0] / actual_sum[2]}")
    

    print(f"\ntraining settings: \nepochs: {FF.EPOCHS}\nbatch size: {FF.BATCH_SIZE}\nlearning rate:{FF.LEARNING_RATE}")
