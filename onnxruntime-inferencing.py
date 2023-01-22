import os
import sys
import time
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2


def preprocess(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.array(Image.fromarray(img)).astype(np.float32)
    # crop 190 380 140 230
    img = img[194:194 + 140, 372:372 + 220]
    img /= 255.
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


class_mapping = ['healthy', 'misalignment', 'rotor damage']
path_res = 'resnet10t.pth'

path_rotor = 'data_rpi/rotor'
path_healthy = 'data_rpi/healthy'
path_misal = 'data_rpi/misalignment'

session = ort.InferenceSession(
    "resnet10t2.onnx",
    providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
rotor_correct = 0
healthy_correct = 0
misalignment_correct = 0
tik_total = time.time()

rotor_tensors = []
for image in os.listdir(path_rotor):
    img = cv2.imread(path_rotor+'/'+image)
    img_prep = preprocess(img)
    rotor_tensors.append(img_prep)

healthy_tensors = []
for image in os.listdir(path_healthy):
    img = cv2.imread(path_healthy+'/'+image)
    img_prep = preprocess(img)
    healthy_tensors.append(img_prep)

misalignment_tensors = []
for image in os.listdir(path_misal):
    img = cv2.imread(path_misal+'/'+image)
    img_prep = preprocess(img)
    misalignment_tensors.append(img_prep)

tik_inferencing = time.time()
for i in range(len(rotor_tensors)):
    prediction = session.run(
        [label_name], {input_name: rotor_tensors[i].astype(np.float32)})[0]
    prediction_index = prediction[0].argmax(0)
    predicted_class = class_mapping[prediction_index]
    if predicted_class == 'rotor damage':
        rotor_correct+=1
    # print(predicted_class)

for i in range(len(healthy_tensors)):
    prediction = session.run(
        [label_name], {input_name: healthy_tensors[i].astype(np.float32)})[0]
    prediction_index = prediction[0].argmax(0)
    predicted_class = class_mapping[prediction_index]
    if predicted_class == 'healthy':
        healthy_correct+=1
    # print(predicted_class)

for i in range(len(misalignment_tensors)):
    prediction = session.run(
        [label_name], {input_name: misalignment_tensors[i].astype(np.float32)})[0]
    prediction_index = prediction[0].argmax(0)
    predicted_class = class_mapping[prediction_index]
    if predicted_class == 'misalignment':
        misalignment_correct+=1
    # print(predicted_class)
tok_inferencing = time.time()
tok_total = time.time()

print(f'Accuracy for healthy class: [{healthy_correct}/40] - {healthy_correct / 40 * 100}%')
print(f'Accuracy for rotor damage class: [{rotor_correct}/40] - {rotor_correct / 40 * 100}%')
print(f'Accuracy for misalignment class: [{misalignment_correct}/40] - {misalignment_correct / 40 * 100}%')
print()
print('Total inferencing time: ', round(tok_inferencing - tik_inferencing, 4), '[s]')
print('Total preprocessing and inferencing time: ', round(tok_total - tik_total, 4), '[s]')
print()
print(f'Average inferencing time per image: {round((tok_inferencing - tik_inferencing)/120, 4)}[s]')
print(f'FPS: {1/((tok_inferencing - tik_inferencing)/120)}')
