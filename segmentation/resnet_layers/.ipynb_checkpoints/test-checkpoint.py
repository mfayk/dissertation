import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
#from torchvision.models
import sys
# Assuming you have a custom dataset class for Cityscapes
from datasets import CityscapesDataset
from torchvision import transforms
from tqdm import tqdm
from model import ResNet
from model import ResBottleneckBlock
from model import ResBlock
from datasets import CityscapesDataset
import sys
from torchsummary import summary



# resnet18
resnet18 = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet18, (3, 224, 224))

# resnet34
resnet34 = ResNet(3, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=1000)
resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet34, (3, 224, 224))

# resnet50
resnet50 = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=1000)
resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet50, (3, 224, 224))

# resnet101
resnet101 = ResNet(3, ResBottleneckBlock, [3, 4, 23, 3], useBottleneck=True, outputs=1000)
resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet101, (3, 224, 224))

# resnet152
resnet152 = ResNet(3, ResBottleneckBlock, [3, 8, 36, 3], useBottleneck=True, outputs=1000)
resnet152.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet152, (3, 224, 224))
