import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import Cityscapes
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

import torch.nn.functional as F
import sys
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models import resnet18
# Assuming you have a custom dataset class for Cityscapes
from datasets import CityscapesDataset
from torchsummaryX import summary
from custom import resnet18_segmentation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        'model_num':[],
        'LR':[],
        'size':[]})


# Data transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to a consistent size
    transforms.ToTensor(),
])


error_bound = ['1E-0','1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']
#error_bound = ['05', '06', '08', '09', '11', '12', '1E-1']
#error_names = ['0.05', '0.06', '0.07', '0.08', '0.09', '0.11', '0.12']
model_nums = ['18', '101','50','end']
LR = ['0.0001','0.0005','0.001','0.0025','0.00005']
LRN = ['0001','0005','001','0025','00005']
size = ['0','1','2','3','4','5']


z=0
while(z < 6):
    print('z')
    print(z)
    y=0
    while(y < 5):
        print('y')
        print(y)
        k = 0
        while(k < 3):
            print('k')
            print(k)
            j = 0
            while(j < 7):
                print('j')
                print(j)
                model_num = model_nums[k]
                flag_error = 0
                while flag_error<10:
                    #print('flag error')
                    #print(flag_error)
                    #model_name = "/project/jonccal/fthpc/mfaykus/NN_models/resnet" + str(model_num) + "_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
                    #model_name = "/scratch/mfaykus/dissertation/NN_models/resnet_cityscapes_pat_" + str(error_bound[j]) + "_1024_run_" + str(flag_error) + ".pth"
                    model_name = "/project/jonccal/fthpc/mfaykus/NN_models/cityscape_resnet/sz3/19class/resnet" + str(model_num) + "_cityscapes_" + str(error_bound[j]) + "_size" + str(size[z]) + "_LR" + str(LR[y]) + "_run" + str(flag_error) + ".pth"

                    #model_name = "/scratch/mfaykus/dissertation/NN_models/compressed/sz3_fixed/resnet" + str(model_num) + "_cityscapes_" + str(error_bound[j]) + "_size" + str(size[z]) + "_LR" + str(LR[y]) + "_run" + str(flag_error) + ".pth"
                    #model_name = "/scratch/mfaykus/dissertation/NN_models/compressed/sz3_finetuned/resnet" + str(model_num) + "_cityscapes_" + str(error_bound[j]) + "_size" + str(size[z]) + "_LR" + str(LRN[y]) + "_run" + str(flag_error) + "_test.pth"
                    
                    
                    #data_path = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2'
                    data_path = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/" + str(error_bound[j]) + "/"
                    #data_path = '/scratch/mfaykus/dissertation/NN_models/compressed/sz3_fixed/resnet'
                    
                    
                    #data_path = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/sz3_finetuned/" + str(error_bound[j]) + "/"
                    print(data_path)
                    if str(error_bound[j]) == "1E-0":
                        data_path = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2"
                    
                    #data_path = '/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/sz3_fixed/'
                    
                    #if str(error_bound[j]) == '1E-0':
                    #    data_path = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2"

                    print(model_name)
                    isFile = os.path.isfile(model_name)

                    if isFile == False:
                        print("cant find file")
                        flag_error += 1
                        continue

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



                    #model = segmentation.deeplabv3_resnet50(num_classes=num_classes)
                    model.load_state_dict(torch.load(model_name))
                    model = model.to(device)
                    model.eval()

                    #root1 = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/zfp/1E-" + str(j) + "/"

                    print(data_path)
                    val_dataset = Cityscapes(root=data_path, split="val")
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)




                    predictions = []
                    targets = []

                    with torch.no_grad():
                        #for a,b,c in val_loader:
                            #print(c)
                        for inputs, labels, image in val_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = model(inputs)



                            if model_num == "18":
                                _, predicted = torch.max(outputs, dim=1)
                            else:
                                _, predicted = torch.max(outputs['out'], dim=1)
                                
                            #pred_labels = torch.argmax(outputs['out'], dim=1)

                            #print("Unique predicted classes:", torch.unique(pred_labels))
                            #print("Unique target classes:", torch.unique(labels))
                            



                            predictions.append(predicted.cpu().numpy())
                            #targets.append(labels.numpy())
                            targets.append(labels.cpu().numpy())


                    predictions = np.concatenate(predictions, axis=0)
                    targets = np.concatenate(targets, axis=0).astype(np.int64)

                    #Ensure targets and predictions are 1D arrays
                    targets = targets.flatten()
                    predictions = predictions.flatten()

                    #print("Unique values in targets:", np.unique(targets))
                    #print("Unique values in predictions:", np.unique(predictions))
                    
                    mask = targets != 0
                    targets = targets[mask]
                    predictions = predictions[mask]


                    #confusion_mat = confusion_matrix(targets, predictions)
                    confusion_mat = confusion_matrix(targets, predictions, labels=np.arange(1, num_classes))  # Exclude 0


                    # Calculate IoU for each class
                    #iou_per_class = np.diag(confusion_mat) / (confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat))
                    
                    denominator = confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat)
                    iou_per_class = np.divide(np.diag(confusion_mat), denominator, out=np.zeros_like(denominator, dtype=float), where=denominator!=0)


                    # Calculate mean IoU
                    mean_iou = np.nanmean(iou_per_class)


                    #print("iou_per_class")

                    #i=0
                    #while i < num_classes:
                    #    print(iou_per_class[i])
                    #    i+=1



                    #print(f"Mean IoU: {mean_iou}")    

                    #classes = ['unlabeled', 'road', 'sidewalk', 'building','wall', 'fence', 'pole', 'traffic light','traffic sign', 'vegetation', 'terrain', 'sky','person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']

                    #i=0
                    #while i < 20:
                        #df.loc[len(df)] = ['ResNet', 'sz3', error_bound[j], '10', mean_iou, classes[i], iou_per_class[i], flag_error, model_nums[k], LR[y], size[z]]
                        
                    classes = ['road', 'sidewalk', 'building','wall', 'fence', 'pole', 'traffic light','traffic sign', 'vegetation', 'terrain', 'sky','person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']    
                    
                    for i in range(19):
                        df.loc[len(df)] = ['ResNet', 'sz3', error_bound[j], '10', mean_iou,
                        classes[i], iou_per_class[i], flag_error,
                        model_nums[k], LR[y], size[z]]
                        df.to_csv('resnet_training.csv')
                        i+=1

                    flag_error += 1
                j = j + 1
            k = k + 1
        y = y + 1
    z = z + 1




df.to_csv('resnet_training.csv')