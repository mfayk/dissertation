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
print(device)

def plot_gradient_flow(namesss, model):
    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []

    #print("before")
    # Loop through model parameters to gather gradients
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu())  # Move to CPU
            max_grads.append(param.grad.abs().max().cpu())   # Move to CPU
            min_grads.append(param.grad.abs().min().cpu())   # Move to CPU
    
    #print("after")
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ave_grads, "b-", label="Mean Gradient")
    plt.plot(max_grads, "g-", label="Max Gradient")
    plt.plot(min_grads, "r-", label="Min Gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads))  # optional: adjust as needed

    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.legend()
    plt.grid(True)

    names = "/scratch/mfaykus/outputs" + str(namesss) + ".png"
    #print(names)
    plt.savefig(names)
    plt.close()


def monitor_gradients(model):
    for name, param in model.named_parameters():
        if "conv" in name and param.requires_grad:
            print(f"Layer: {name} | Grad: {param.grad.abs().mean()}")


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




# Hyperparameters
num_classes = 34  # Cityscapes has 19 classes
batch_size = 32
LR =  str(sys.argv[3])
learning_rate = float(LR)
LR = 0.001
#learning_rate = 0.001
num_epochs = 100


resize = int(str(sys.argv[4]))
image_resizing_x = [512, 512, 768, 768, 1024, 1024]
image_resizing_y = [512, 1024, 768, 1536, 1024, 2048]






# Data transformations
transform = transforms.Compose([
    transforms.Resize((int(image_resizing_x[resize]), int(image_resizing_y[resize]))),  # Resize images to a consistent size
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to match the size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#target_transform = transforms.Compose([
#    transforms.Resize((int(image_resizing_x[resize]), int(image_resizing_y[resize]))),  # Resize segmentation masks
#    transforms.ToTensor()
#])


error =  str(sys.argv[1])
model_num = str(sys.argv[2])




#path_name = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/compressed/zfp/" + error + "/"
#path_name = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/sz3_fixed/" + error + "/"
path_name = "/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/sz3_finetuned/" + error + "/"
print(path_name)
if error == "1E-0":
    path_name = "/project/jonccal/fthpc/mfaykus/datasets/cityscapes2"


    
# Cityscapes dataset
train_dataset = CityscapesDataset(root=path_name, split="train", augment=transform)
val_dataset = CityscapesDataset(root=path_name, split="val", augment=transform)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define early stopping parameters
patience = 100
best_val_loss = float('inf')
current_patience = 0
#model_num = 18
print(model_num)

flag_error = 0
while flag_error<10:

    print("LR")
    print(LR)
    print("error")
    print(error)
    print("model_num")
    print(model_num)
    print("path")
    path_name = "/scratch/mfaykus/dissertation/NN_models/compressed/sz3_finetuned/resnet" + str(model_num) + "_cityscapes_" + error + "_size" + str(resize) + "_LR" + str(sys.argv[3]) + "_run" + str(flag_error) + "_test.pth"
    print(path_name)

    # DeepLabV3 with ResNet backbone
    
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
    #model = resnet18_segmentation(num_classes=num_classes)
    
    #model = model.to(device)
    #tensor = torch.zeros((1, 3, 1024, 1024)).to(device)
    #model = model.to(device, torch.zeros((1, 3, 1024, 1024)).to(device))
    model = model.to(device)
    tensor = torch.zeros((1, 3, 1024, 1024)).to(device)

    #summary(model)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    

    

    validation_interval = 1

    # Training loop
    nameesssss = 0
    for epoch in range(num_epochs):
        #print("start training")
        model.train()
        #print("model train")
        train_loss = 0.0
        for inputs, targets, mask in train_loader:
            #imagergb, labelmask, labelrgb
            #print("before device")
            inputs, targets = inputs.to(device), targets.to(device)
            #print("moved to device")
            targets = targets.squeeze(1).long()

            optimizer.zero_grad()

            #print("opmizer")
            # Forward pass
            outputs = model(inputs)
            
            #print(outputs)
            #print("forward pass")
            

            #print("Output shape:", outputs['out'].shape)
            #print("Targets shape:", targets.shape)

            # Calculate loss
            if model_num == "18":
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs['out'], targets)

            #print("initalize back prop")
            # Backward pass and optimization
            loss.backward()
            
            plot_gradient_flow(str(error) + '_' + str(nameesssss),model)
            nameesssss = nameesssss + 1
            optimizer.step()
            
            train_loss += loss.item()# * inputs.size(0)
            #print("getting loss")
        #print("epoch")
    
        train_loss /= len(train_loader.dataset)


        
        print("validation")
            # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, mask in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1).long()
                outputs = model(inputs)
                if model_num == "18":
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs['out'], targets)
                
                pred_labels = torch.argmax(outputs['out'], dim=1)
                print("Unique predicted classes:", torch.unique(pred_labels))
                print("Unique target classes:", torch.unique(targets))
                val_loss += loss.item() * inputs.size(0)
                
                
                
            #print(val_loader.dataset)
            val_loss /= len(val_loader.dataset)

        # Check if validation loss has improved
        if val_loss < 200:
            best_val_loss = val_loss
            current_patience = 0
            # Save the model if needed
            path_name = "/scratch/mfaykus/dissertation/NN_models/compressed/sz3_finetuned/resnet" + str(model_num) + "_cityscapes_" + error + "_size" + str(resize) + "_LR" + str(sys.argv[3]) + "_run" + str(flag_error) + "_test.pth"
            #path_name = "bestmodel.pth"
            print(path_name)
            print('before save')
            torch.save(model.state_dict(), path_name)
            print("saved")
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
    
