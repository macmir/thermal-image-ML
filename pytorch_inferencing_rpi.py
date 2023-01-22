import os

import torch
import timm
from PIL import Image
from torchvision import transforms, models
import cv2
import matplotlib.pyplot as plt

def preprocess_tensor(path):
	img = cv2.imread(path)
	img = cv2.normalize(img, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
	img = Image.fromarray(img)
	img = transforms.functional.crop(img, 190, 380, 140, 230)
	to_tensor = transforms.ToTensor()
	img = to_tensor(img)
	return img

class_mapping = ['healthy', 'misalignment', 'rotor damage']
path_res = 'saved_models/resnet10t_best.pth'
path_rotor = 'captum_images/rotor.png'
path_healthy = 'captum_images/healthy.png'
path_misal = 'captum_images/misalignment.png'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

resnet10t = timm.create_model('resnet10t', pretrained=True, num_classes=3)
# print(resnet10t)

model_resnet = resnet10t
model_resnet.load_state_dict(torch.load(path_res, map_location=torch.device('cpu')))

rotor_tensors = []
for image in os.listdir(path_rotor):
    img = cv2.imread(path_rotor+'/'+image)
    img_prep = preprocess_tensor(img)
    rotor_tensors.append(img_prep)

# input_test_rotor = preprocess_tensor(path_rotor)
input_test_misal = preprocess_tensor(path_misal)
input_test_healthy = preprocess_tensor(path_healthy)

# target_test_rotor = torch.tensor(int(2))
target_test_misal = torch.tensor(int(1))
target_test_healthy = torch.tensor(int(0))

model_resnet.eval()
with torch.no_grad():
   input_test_rotor = torch.unsqueeze(input_test_rotor, 0) # adds another dimension (N C H W)
   input_test_rotor = input_test_rotor.to(device) # send data to GPU or CPU
   predictions_rotor = model_resnet(input_test_rotor)
   predicted_index_rotor = predictions_rotor[0].argmax(0)
   predicted_test_rotor = class_mapping[predicted_index_rotor]
   expected_test_rotor = class_mapping[target_test_rotor]

   input_test_misal = torch.unsqueeze(input_test_misal, 0) # adds another dimension (N C H W)
   input_test_misal = input_test_misal.to(device) # send data to GPU or CPU
   predictions_misal = model_resnet(input_test_misal)
   predicted_index_misal = predictions_misal[0].argmax(0)
   predicted_test_misal = class_mapping[predicted_index_misal]
   expected_test_misal = class_mapping[target_test_misal]

   input_test_healthy = torch.unsqueeze(input_test_healthy, 0) # adds another dimension (N C H W)
   input_test_healthy = input_test_healthy.to(device) # send data to GPU or CPU
   predictions_healthy = model_resnet(input_test_healthy)
   predicted_index_healthy = predictions_healthy[0].argmax(0)
   predicted_test_healthy = class_mapping[predicted_index_healthy]
   expected_test_healthy = class_mapping[target_test_healthy]


if(predicted_test_rotor == expected_test_rotor):
  print("Correctly predicted Rotor damaged!")

if(predicted_test_misal == expected_test_misal):
  print("Correctly predicted Misalignment!")

if(predicted_test_healthy == expected_test_healthy):
  print("Correctly predicted Healthy!")


# Plots images
fig = plt.figure()
fig.set_figheight(3)
fig.set_figwidth(14)

plt.subplot(1, 3, 1)
plt.imshow(preprocess_img(path_rotor))
plt.xlabel("predicts: " + predicted_test_rotor)

plt.subplot(1, 3, 2)
plt.imshow(preprocess_img(path_healthy))
plt.xlabel("predicts: " + predicted_test_healthy)

plt.subplot(1, 3, 3)
plt.imshow(preprocess_img(path_misal))
plt.xlabel("predicts: " + predicted_test_misal)

fig.suptitle('RPi Deployment [Torch Inferencing]')
plt.show()
