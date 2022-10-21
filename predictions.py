import torch
import train
import clutchDataset


class_mapping = [
    "healthy",
    "misalignment",
    "rotor damage"
]
good_prediction = 0
bad_prediction = 0
total = 0
def predict(model, input, target, class_mapping):
    model.eval()    # call it every time you want to make an inference
    with torch.no_grad():
        #print(input.size())


        predictions = model(input) # Tensor(1, 10) - 1-number of inputs, 10-number of classes
        # Tensor(1, 10) -> [[0.1, 0.1, 0.2 ... 0.6]] - sums up to 1.0 as we used softmax
        predicted_index = predictions[0].argmax(0) # take first max value
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    # load back the model
    feed_forward_net = train.FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load validation dataset
    validation_data = clutchDataset.valid_dataset

    # get a sample from the validation dataset for inference
    for i in range(len(validation_data)):
        input, target = validation_data[i][0], validation_data[i][1]
    # make an inference
        predicted, expected = predict(feed_forward_net, input, target, class_mapping)
     #   print(f"Predicted: '{predicted}', expected: '{expected}'")
        if predicted == expected:
            good_prediction += 1
            total +=1
        else:
            bad_prediction +=1
            total +=1

    print(f"Good predictions: {good_prediction}\n Bad predictions: {bad_prediction}\n Total: {total}")