import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CityscapesDataset
from torchvision.models import resnet18
import torch.optim as optim
from torchvision.models import segmentation
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import csv
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import segmentation
from torchvision.models import resnet18
import torch.nn.functional as F
import sys
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models import resnet18
# Assuming you have a custom dataset class for Cityscapes
from datasets import CityscapesDataset
from torchsummaryX import summary
from custom import resnet18_segmentation
import torch
from torchvision import transforms
from PIL import Image

import shap
import numpy as np


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super(CustomDeepLabV3, self).__init__()
        # Load ResNet-18 and remove the fully connected layer
        resnet = resnet18()
        
        # Remove the fully connected layer and avg pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add a simple segmentation head using Conv layers + Upsampling
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Pass input through ResNet-18 backbone
        x = self.backbone(x)
        
        # Apply segmentation head
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Upsample to original image size (assuming input size is divisible by 32)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)
        
        # Output pixel-wise class predictions
        x = self.conv3(x)
        return x

num_classes = 20

df = pd.DataFrame({
        'Model':[],
        'Compressor':[],
        'error_bound':[],
        'epochs':[],
        'miou':[],
        'class':[],
        'iou':[],
        'run':[],
         'model_num':[]})


# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),
])


error_bound = ['base','1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']
model_nums = ['50','18', '101','end']
model_num = '50'
flag_error = 0

model_name = "/project/jonccal/fthpc/mfaykus/NN_models/cityscape_resnet/sz3/19class/resnet" + str(model_nums[0]) + "_cityscapes_pat_" + str(error_bound[0]) + "_1024_run_" + str(flag_error) + ".pth"
            #model_name = "/scratch/mfaykus/dissertation/NN_models/resnet_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2'
#data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/leftImg8bit/val/frankfurt_000000_000294_leftImg8bit.png'
#data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/1E-5/leftImg8bit/val/frankfurt_000000_000294_leftImg8bit.png'

if model_num == "18":
    model = CustomDeepLabV3(num_classes=num_classes)
elif model_num == "34":
    model = segmentation.deeplabv3_resnet34(num_classes=num_classes)
elif model_num == "50":
    model = segmentation.deeplabv3_resnet50(num_classes=num_classes)
elif model_num == "101":
    model = segmentation.deeplabv3_resnet101(num_classes=num_classes)
elif model_num == "152":
    model = segmentation.deeplabv3_resnet152(num_classes=num_classes)

model.load_state_dict(torch.load(model_name))
model.eval()

cityscapes = CityscapesDataset(root=data_path, split="val", augment=transform)
data_loader = torch.utils.data.DataLoader(cityscapes, batch_size=1, shuffle=False)




dummy_image = np.random.rand(1, 128, 256, 3)  # Dummy input in HWC format
dummy_tensor = torch.tensor(np.transpose(dummy_image, (0, 3, 1, 2)), dtype=torch.float32)  # Convert to CHW

with torch.no_grad():
    output = model(dummy_tensor)
    print(output)

def model_predict(images):
    # Assuming images are in [batch_size, height, width, channels], convert to [batch_size, channels, height, width]
    images = np.transpose(images, (0, 3, 1, 2))  # Transpose from (batch_size, height, width, channels) to (batch_size, channels, height, width)
    images = torch.tensor(images, dtype=torch.float32)

    with torch.no_grad():
        output = model(images)  # Output should have shape [batch_size, num_classes, height, width]
        probabilities = torch.softmax(output, dim=1)  # Apply softmax to get class probabilities
        return probabilities.cpu().numpy()

background = np.random.rand(10, 128, 256, 3)  # Shape [batch_size, height, width, channels]
background = np.transpose(background, (0, 3, 1, 2))  # Convert to [batch_size, channels, height, width]




print(f"Background shape: {background.shape}")
print(f"Input shape: {dummy_image.shape}")

explainer = shap.KernelExplainer(model_predict, background)

# Test the explainer with a dummy image
dummy_image = np.random.rand(1, 128, 256, 3)  # [batch_size, height, width, channels]
dummy_image = np.transpose(dummy_image, (0, 3, 1, 2))  # Convert to [batch_size, channels, height, width]

# Get the SHAP values
shap_values = explainer.shap_values(dummy_image)

# Visualize the SHAP values
shap.image_plot(shap_values, dummy_image)

        
exit()

# Create SHAP explainer
explainer = shap.Explainer(model_predict, shap.maskers.Image("inpainting", (128, 256, 3)))

dummy_image = np.random.rand(1, 128, 256, 3)  # Replace with actual image shape
predictions = model_predict(dummy_image)
print(predictions.shape) 
explainer = shap.Explainer(model_predict, shap.maskers.Image("inpainting", (256, 256, 3)))

# Compute SHAP values
dummy_image = np.random.rand(1, 256, 256, 3)  # Replace with real images
shap_values = explainer(dummy_image)

# Visualize SHAP values
shap.image_plot(shap_values, dummy_image)

print("here")
exit()


# Iterate through dataset
for inputs, labels, images in data_loader:
    # Compute SHAP values
    images = np.transpose(images, (0, 3, 1, 2))  # Convert to PyTorch format
    images = torch.tensor(images, dtype=torch.float32)
    shap_values = explainer(inputs)

    # Plot SHAP values
    shap.image_plot(shap_values, inputs.numpy())
    break  # Remove this to loop through more samples
    
    


    
    # Extract SHAP values for a specific class (e.g., class index 10)
class_index = 10
shap_for_class = shap_values[:, class_index, :, :]

# Aggregate SHAP values (optional)
shap_aggregated = np.sum(shap_for_class, axis=0)

# Plot the SHAP values for the class
plt.imshow(shap_aggregated, cmap='viridis')
plt.title(f"SHAP Values for Class {class_index}")
plt.colorbar()
plt.savefig('sharp0.png')
plt.close()
