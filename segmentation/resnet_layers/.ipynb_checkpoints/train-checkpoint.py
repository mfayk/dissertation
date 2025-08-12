import torch
import sys
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import segmentation
from torchvision.models import resnet18
import sys
# Assuming you have a custom dataset class for Cityscapes
from datasets import Cityscapes
from torchsummaryX import summary
#from custom import resnet18_segmentation

# Hyperparameters
num_classes = 20  # Cityscapes has 19 classes
batch_size = 32
learning_rate = 0.001
num_epochs = 100


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Hyperparameters
num_classes = 20  # Cityscapes has 19 classes
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# Data transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to a consistent size
    transforms.ToTensor(),
])

error =  str(sys.argv[1])
model_num = str(sys.argv[2])

path_name = "/scratch/mfaykus/dissertation/scratch_backup/dissertation/datasets/cityscapes2/compressed/" + error + "/"

if error == "base":
    path_name = "/scratch/mfaykus/dissertation/scratch_backup/dissertation/datasets/cityscapes2"

path_name = "/scratch/mfaykus/dissertation/scratch_backup/dissertation/datasets/cityscapes/"    
    
# Cityscapes dataset
train_dataset = Cityscapes(root=path_name, split="train", transform=transform)
val_dataset = Cityscapes(root=path_name, split="val", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Define early stopping parameters
patience = 10
best_val_loss = float('inf')
current_patience = 0
#model_num = 18
print(model_num)

flag_error = 0
while flag_error<10:

    # Load a pre-trained ResNet-18 backbone
    resnet18 = models.resnet18()

    # Load DeepLabV3 with ResNet-50 and swap the backbone with ResNet-18
    deeplabv3 = models.segmentation.deeplabv3_resnet50()

    # Replace the backbone with ResNet-18
    deeplabv3.backbone = nn.Sequential(*list(resnet18.children())[:-2])  # Replace the backbone up to layer4 (final conv layer)
    deeplabv3.classifier[4] = nn.Conv2d(256, deeplabv3.classifier[4].out_channels, kernel_size=(1, 1), stride=(1, 1))

    # Send model to GPU if available
    deeplabv3 = deeplabv3.to(device)




    tensor = torch.zeros((1, 3, 256, 256)).to(device)

    #summary(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(deeplabv3.parameters(), lr=learning_rate)


    validation_interval = 1

    
    # Training loop
    for epoch in range(num_epochs):
        deeplabv3.train()
        train_loss = 0.0
        print("here")
        for (inputs, targets) in train_loader:
            #imagergb, labelmask, labelrgb
            print("here")
            inputs, targets = inputs.to(device), targets.to(device)
            print("here")
            targets = targets.squeeze(1).long()
            print("here")
            
            optimizer.zero_grad()

            
            print(inputs.shape) 
            print(targets.shape)
            print(mask.shape)
            # Forward pass
            outputs = deeplabv3(inputs)
            
            #print(outputs)


            #print("Output shape:", outputs['out'].shape)
            #print("Targets shape:", targets.shape)

            # Calculate loss
            loss = criterion(outputs['out'], targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
    
        train_loss /= len(train_loader.dataset)
        
        
            # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, mask in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1).long()
                outputs = model(inputs)
                loss = criterion(outputs['out'], targets)
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
            # Save the model if needed
            path_name = "/scratch/mfaykus/dissertation/NN_models/resnet" + str(model_num) + "_cityscapes_pat_" + error + "_1024_run_" + str(flag_error) + ".pth"
            #path_name = "bestmodel.pth"
            torch.save(model.state_dict(), path_name)
        else:
            current_patience += 1
            if current_patience == patience:
                print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
                break

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        
        
        



        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Save the trained model
    #torch.save(model.state_dict(), '/zfs/fthpc/mfaykus/NN_models/cityscape_resnet/sz3/19class/deeplab_resnet_cityscapes_base.pth')

    #path_name = "/scratch/mfaykus/dissertation/NN_models/deeplab_resnet_cityscapes_100_" + error + "_1024_" + str(flag_error) + ".pth"
    #path_name = "/scratch/mfaykus/dissertation/NN_models/deeplab_resnet_cityscapes_100_base_1024_run_" + str(flag_error) + ".pth"
    # Save the trained model
    #torch.save(model.state_dict(), path_name)

    flag_error += 1
    
