import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import LayerGradCam
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerAttribution
from captum.attr import GuidedGradCam
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
import CNN
import cv2
import clutchDataset
import timm


model_no = 4

if model_no == 1:
    model = CNN.CNN()
    state_dict = torch.load('saved_models/CNN_3ch_cropped.pth')
    model.load_state_dict(state_dict)
    model.eval()

elif model_no == 2:
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT, in_channels=3, n_classes=3)
    # model = model.to(device)
    state_dict = torch.load('saved_models/efficientnet_v2_s_1.pth')
    model.load_state_dict(state_dict)
    model.eval()

elif model_no == 3:
    model = torchvision.models.resnet34(pretrained=True)
    # model = model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    # model.fc = model.fc.cuda()
    state_dict = torch.load('saved_models/resnet34.pth')
    model.load_state_dict(state_dict)
    model.eval()

elif model_no == 4:
    model = timm.create_model('resnet10t', pretrained=True, num_classes=3)
    state_dict = torch.load('saved_models/resnet10t_best.pth')
    model.load_state_dict(state_dict)
    model.eval()


# converting .png to .jpg
png_img = cv2.imread('captum_images/rotor_test.png')
cv2.imwrite('captum_images/modified_img.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

test_img = Image.open('captum_images/modified_img.jpg')
test_img_data = np.asarray(test_img)
img_norm = cv2.normalize(test_img_data, None, 0, 255, cv2.NORM_MINMAX)
image_cv2 = Image.fromarray(img_norm)
# image_cv2 = transforms.functional.crop(image_cv2, 194, 285 , 140, 320)
image_cv2 = transforms.functional.crop(image_cv2, 194, 372, 140, 220)

# plt.imshow(image_cv2, cmap="gray")
# plt.show()


width = clutchDataset.width
height = clutchDataset.height
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((height, width))
    ]
    )


# img = Image.open('modified_img.jpg')

# transformed_img = transform(img)

# input2 = transform(transformed_img)


# image_cv2 = cv2.imread('modified_img.jpg')
# image = Image.fromarray(img_norm)
tensor = transform(image_cv2)
tensor = tensor[None, :]
# print(tensor.shape)

#         img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         image = Image.fromarray(img_norm)


# transform = transforms.Compose([transforms.ToTensor()])

# tensor = transform(img)

# testloader = CNN.test_loader
# dataiter = iter(testloader)
# input, labels = dataiter.next()
labels = torch.tensor([2])

# print(input.shape)
output = model(tensor)
print(output)
print(output.size)

predicted_index = output[0].argmax(0)

print("Predicted index: ", predicted_index)


#Create IntegratedGradients object and get attributes
integrated_gradients = IntegratedGradients(model)
#Request the algorithm to assign our output target to
attributions_ig = integrated_gradients.attribute(tensor, target=predicted_index, n_steps=300, method="riemann_left")
#
# default_cmap = LinearSegmentedColormap.from_list('custom blue',
#                                                  [(0, '#ffffff'),
#                                                   (0.25, '#000000'),
#                                                   (1, '#000000')], N=256)
# # use visualize_image_attr helper method for visualization to show the #original image for comparison
# _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
#                              np.transpose(tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
#                              method='blended_heat_map',
#                              cmap="Reds",
#                              sign='negative',
#                              outlier_perc=2)

noise_tunnel = NoiseTunnel(integrated_gradients)

attributions_ig_nt = noise_tunnel.attribute(tensor, nt_samples=10, nt_type='smoothgrad_sq', target=predicted_index)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      img_norm,
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap="gnuplot",
                                      show_colorbar=True)

# guided_gc = GuidedGradCam(model, model.)

                                      
