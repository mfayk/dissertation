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
import torch

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
    transforms.Resize((1024, 1024)),  # Resize images to a consistent size
    transforms.ToTensor(),
])


error_bound = ['base','1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']
model_nums = ['50','18', '101','end']
model_num = '50'
flag_error = 0

model_name = "/project/jonccal/fthpc/mfaykus/NN_models/cityscape_resnet/sz3/19class/resnet" + str(model_nums[0]) + "_cityscapes_pat_" + str(error_bound[0]) + "_1024_run_" + str(flag_error) + ".pth"
            #model_name = "/scratch/mfaykus/dissertation/NN_models/resnet_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
#data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2'
#data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/leftImg8bit/val/frankfurt_000000_000294_leftImg8bit.png'
data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/1E-1/leftImg8bit/val/frankfurt_000000_000294_leftImg8bit.png'

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


transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to a tensor (values in range [0, 1])
    # You can add other transformations like normalization here if needed
])

image = Image.open(data_path)
input_image = transform(image)

input_image = input_image.unsqueeze(0)

# Assume 'model' is your trained ResNet model, and 'input_image' is the input tensor
input_image.requires_grad_()  # Ensure the input requires gradients

# Forward pass
output = model(input_image)
#print(output)
logits = output['out']  # or output['output'], depending on your model's output structure
logits = logits.squeeze()

# Loop through each class
num_classes = logits.shape[0]  # Number of classes in the output
for class_idx in range(num_classes):
    # Zero all gradients
    model.zero_grad()

    # Select the logits for the current class, sum over all pixels
    class_logit = logits[class_idx].sum()  # Sum of logits for all pixels of the current class

    # Perform backward pass to compute gradients w.r.t the input
    class_logit.backward(retain_graph=True)

    # Get the saliency map (maximum of the absolute gradients)
    saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)

    # Plot the saliency map for the current class
    plt.imshow(saliency.squeeze().cpu(), cmap='hot')
    plt.axis('off')
    plt.title(f"Saliency Map for Class {class_idx}")
    plt.savefig(f"saliency_1E1_class_{class_idx}.png")  # Save the plot for each class
    plt.close()