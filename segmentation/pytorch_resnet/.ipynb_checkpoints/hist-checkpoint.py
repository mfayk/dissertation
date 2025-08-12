#histogram

#a and b and diff



#next week:
#    kinda of plots I want to make
#    make slides
#    x,y and what it looks like to analuzy train data
    
    
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
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hook setup
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Visualization function for activations
def plot_activations(name, layer, num_cols=1, num_activations=16):
    num_kernels = layer.shape[1]
    fig, axes = plt.subplots(nrows=(num_activations + num_cols - 1) // num_cols, ncols=num_cols, figsize=(64, 64))
    #print("before")
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            ax.imshow(layer[0, i].cpu().numpy(), cmap='twilight')
            ax.axis('off')
    #print("after")
    #plt.tight_layout()
    #plt.show()
    plt.savefig(name)
    plt.close()


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


error_bound = ['base','1E-7','1E-1','1E-6','1E-5','1E-4','1E-3','1E-2']
model_nums = ['50', '18','101','end']

flag = 8
k = -1
j = 0
while(k < 1):
    k = k + 1
    flag = 1
    
    j = 0
    while(j < 8):
        model_num = model_nums[k]
        flag_error = 0
        while flag_error<1:
            #model_name = "/scratch/mfaykus/dissertation/NN_models/resnet" + str(model_num) + "_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
            model_name = "/project/jonccal/fthpc/mfaykus/NN_models/cityscape_resnet/sz3/19class/resnet" + str(model_num) + "_cityscapes_pat_" + 'base' + "_1024_run_" + str(flag_error) + ".pth"
            #model_name = "/scratch/mfaykus/dissertation/NN_models/resnet_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
            #data_path = '/scratch/mfaykus/dissertation/datasets/datasets/cityscapes2'
            data_path = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/" + str(error_bound[j]) + "/"

            if str(error_bound[j]) == "base":
                data_path = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2"

            print(model_name)
            isFile = os.path.isfile(model_name)

            if isFile == False:
                flag_error += 1
                continue

            #print("here")
            # Instantiate the model
            
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
            
            #print(model)
            
            #model = segmentation.deeplabv3_resnet50(num_classes=num_classes)
            model.load_state_dict(torch.load(model_name))
            model.eval()
            
            # Register hooks for specific layers
            feature_maps = {}
            
            model.backbone.layer1[0].conv1.register_forward_hook(get_activation('layer1_conv1'))
            model.backbone.layer2[0].conv1.register_forward_hook(get_activation('layer2_conv1'))
            model.backbone.layer3[0].conv1.register_forward_hook(get_activation('layer3_conv1'))
            model.backbone.layer4[0].conv1.register_forward_hook(get_activation('layer4_conv1'))

            #root1 = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/zfp/1E-" + str(j) + "/"

            val_dataset = CityscapesDataset(root=data_path, split="val", augment=transform)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

            feature_maps = []  # List to store feature maps
            layer_names = []  # List to store layer names
            

            predictions = []
            targets = []

            with torch.no_grad():
                #for a,b,c in val_loader:
                 #   print(c)
                images = 0
                #print("before loader")
                for inputs, labels, image in val_loader:
                    #print("after loader")
                    outputs = model(inputs)
                    
                    # Display a subset of activations
                    print(error_bound[j])
                    name = 'layer1_conv1_' + str(error_bound[j])
                    plot_activations(name, activations['layer1_conv1'], num_cols=4, num_activations=8)
                    name = 'layer2_conv1_' + str(error_bound[j])
                    plot_activations(name, activations['layer2_conv1'], num_cols=4, num_activations=8)
                    name = 'layer3_conv1_' + str(error_bound[j])
                    plot_activations(name, activations['layer3_conv1'], num_cols=4, num_activations=8)
                    name = 'layer4_conv1_' + str(error_bound[j])
                    plot_activations(name, activations['layer4_conv1'], num_cols=4, num_activations=8)
                    

                    
                    
                    if model_num == "18":
                        _, predicted = torch.max(outputs, dim=1)
                    else:
                        _, predicted = torch.max(outputs['out'], dim=1)
                    

                    predictions.append(predicted.cpu().numpy())
                    targets.append(labels.numpy())
                    images += 1
                    #if images == 1:
                        #break
                        

            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0).astype(np.int64)

            # Ensure targets and predictions are 1D arrays
            targets = targets.flatten()
            predictions = predictions.flatten()

            print("Unique values in targets:", np.unique(targets))
            print("Unique values in predictions:", np.unique(predictions))


            confusion_mat = confusion_matrix(targets, predictions)

            # Calculate IoU for each class
            iou_per_class = np.diag(confusion_mat) / (confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat))

            # Calculate mean IoU
            mean_iou = np.nanmean(iou_per_class)


            print("iou_per_class")

            #i=0
            #while i < num_classes:
            #    print(iou_per_class[i])
            #    i+=1



            #print(f"Mean IoU: {mean_iou}")    

            #classes = ['unlabeled', 'road', 'sidewalk', 'building','wall', 'fence', 'pole', 'traffic light','traffic sign', 'vegetation', 'terrain', 'sky','person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']

            #i=0
            #while i < 20:
            #    df.loc[len(df)] = ['unet', 'sz3', error_bound[j], '10', mean_iou, classes[i], iou_per_class[i], flag_error, model_nums[k]]
            #    i+=1


            flag_error += 1
        j+=1
        flag -= 1

    
    



df.to_csv('resnet_train.csv')