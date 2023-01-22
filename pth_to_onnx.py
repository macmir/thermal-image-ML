import torch
import timm
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def preprocess(img):

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    img = np.array(Image.fromarray(img)).astype(np.float32)

    #crop 190 380 140 230
    img = img[194:194+140, 372:372+220]

    img /= 255.

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


model = timm.create_model('resnet10t', pretrained=True, num_classes=3)

state_dict = torch.load('saved_models/resnet10t_best.pth')
model.load_state_dict(state_dict)

model.eval()

dummy_input = torch.randn(1, 3, 140, 220)

input_names = [ "actual_input" ]
output_names = [ "output" ]

torch.onnx.export(model,
                 dummy_input,
                 "resnet10t2.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )


img = cv2.imread('captum_images/healthy_test.png')

img_p = preprocess(img)
print(img_p.shape)

# plt.imshow(img_p[0])
# plt.show()

sess = ort.InferenceSession("resnet10t2.onnx", providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: img_p.astype(np.float32)})[0]
print(pred_onx)
