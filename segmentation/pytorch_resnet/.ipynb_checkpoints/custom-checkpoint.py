import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the BasicBlock for ResNet-18 and ResNet-34
class BasicBlock(nn.Module):
    expansion = 1  # For basic block, the expansion is 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the Bottleneck block for ResNet-50, ResNet-101, ResNet-152
class Bottleneck(nn.Module):
    expansion = 4  # For bottleneck, the expansion is 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the ResNet backbone model
class ResNetBackbone(nn.Module):
    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create layers using the _make_layer function
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4, x3, x2, x1  # Return feature maps for decoder use

# Define the segmentation model using ResNet backbone
class ResNetSegmentation(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ResNetSegmentation, self).__init__()
        self.backbone = backbone

        # Define the upsampling layers for the decoder
        self.upsample1 = nn.ConvTranspose2d(512 * backbone.layer4[0].expansion, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        # Final segmentation output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x4, x3, x2, x1 = self.backbone(x)

        x = self.upsample1(x4)
        x = x + x3  # Skip connection
        
        x = self.upsample2(x)
        x = x + x2  # Skip connection
        
        x = self.upsample3(x)
        x = x + x1  # Skip connection
        
        x = self.upsample4(x)
        
        x = self.final_conv(x)
        return x

# Define functions to create different versions of ResNet for segmentation
def resnet18_segmentation(num_classes=21):
    backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2])
    return ResNetSegmentation(backbone, num_classes)

def resnet34_segmentation(num_classes=21):
    backbone = ResNetBackbone(BasicBlock, [3, 4, 6, 3])
    return ResNetSegmentation(backbone, num_classes)

def resnet50_segmentation(num_classes=21):
    backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
    return ResNetSegmentation(backbone, num_classes)

def resnet101_segmentation(num_classes=21):
    backbone = ResNetBackbone(Bottleneck, [3, 4, 23, 3])
    return ResNetSegmentation(backbone, num_classes)

def resnet152_segmentation(num_classes=21):
    backbone = ResNetBackbone(Bottleneck, [3, 8, 36, 3])
    return ResNetSegmentation(backbone, num_classes)

# Example usage
if __name__ == "__main__":
    # Create a segmentation model
    model = resnet18_segmentation(num_classes=21)  # Change to desired number of classes
    print(model)

    # Create a random input tensor
    x = torch.randn(1, 3, 224, 224)
    # Forward pass through the model
    output = model(x)
    print(output.shape)  # Output should have shape [batch_size, num_classes, H, W]
