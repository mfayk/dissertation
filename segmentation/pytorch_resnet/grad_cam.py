import torch
import torch.nn.functional as F
import cv2
import numpy as np

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

from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_cam_segmentation(model, input_image, target_class, target_layer, target_pixel):
    model.eval()

    def forward_hook(module, input, output):
        global activation
        activation = output

    def backward_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0]

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image)
    target_score = output[0, target_class, target_pixel[0], target_pixel[1]]
    model.zero_grad()
    target_score.backward()

    # Global average pooling of gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight feature maps by gradients
    for i in range(pooled_gradients.size(0)):
        activation[:, i, :, :] *= pooled_gradients[i]

    # Create heatmap
    heatmap = torch.mean(activation, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # Resize heatmap to match input image size
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[3]))
    heatmap = np.uint8(255 * heatmap)

    # Overlay heatmap on original image
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.uint8(255 * img)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()

    return overlay



path_name = "/scratch/mfaykus/dissertation/datasets/datasets/cityscapes2"