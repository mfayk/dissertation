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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 20


df = pd.DataFrame({
        'Model':[],
        'Compressor':[],
        'error_bound':[],
        'epochs':[],
        'miou':[],
        'class':[],
        'iou':[],
        'run':[]})

# Data transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to a consistent size
    transforms.ToTensor(),
])

error_bound = ['base','1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']


j = 8
flag = 0
while j > 0:
    print(j)
    
    
    if(j == 8):
        mod = '/scratch/mfaykus/dissertation/scratch_backup/dissertation/NN_models/deeplab_resnet_cityscapes_100_base_1024.pth'
        root1 = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/'
    else:
        mod = '/scratch/mfaykus/dissertation/scratch_backup/dissertation/NN_models/deeplab_resnet_cityscapes_100_1E-' + str(j) + '_1024.pth'
        root1 = '/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/1E-' + str(j) + '/'

    
    
    #mod = '19class/deeplab_resnet_cityscapes_100_1E' + str(j) + '.pth'
    
    
    # Instantiate the model
    model = segmentation.deeplabv3_resnet50(num_classes=num_classes)
    model.load_state_dict(torch.load(mod))
    model.eval()

    #root1 = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/zfp/1E-" + str(j) + "/"
    
    val_dataset = Cityscapes(root=root1, split="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    flag_error = 0
    while flag_error<10:
    
        predictions = []
        targets = []

        with torch.no_grad():
            #for a,b,c in val_loader:
             #   print(c)
            for inputs, labels, image in val_loader:
                outputs = model(inputs)

                _, predicted = torch.max(outputs['out'], dim=1)

                predictions.append(predicted.cpu().numpy())
                targets.append(labels.numpy())

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

        i=0
        while i < num_classes:
            print(iou_per_class[i])
            i+=1



        print(f"Mean IoU: {mean_iou}")    

        classes = ['unlabeled', 'road', 'sidewalk', 'building','wall', 'fence', 'pole', 'traffic light','traffic sign', 'vegetation', 'terrain', 'sky','person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']

        i=0
        while i < 20:
            df.loc[len(df)] = ['unet', 'sz3', error_bound[flag], '10', mean_iou, classes[i], iou_per_class[i], flag_error]
            i+=1
            
        flag_error += 1
    j-=1
    flag+=1
    
    
    



df.to_csv('resnet_train.csv')