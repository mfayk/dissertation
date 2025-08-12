import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetSegmentation(nn.Module):
    def __init__(self, num_classes, resnet_depth=50, pretrained=True):
        super(ResNetSegmentation, self).__init__()
        
        # Choose ResNet backbone based on depth
        if resnet_depth == 18:
            self.backbone = models.resnet18()
        elif resnet_depth == 34:
            self.backbone = models.resnet34()
        elif resnet_depth == 50:
            self.backbone = models.resnet50()
        elif resnet_depth == 101:
            self.backbone = models.resnet101()
        elif resnet_depth == 152:
            self.backbone = models.resnet152()
        else:
            raise ValueError(f"Unsupported ResNet depth: {resnet_depth}")
        
        # Remove fully connected layer (used for classification)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Segmentation decoder (Upsample + Conv layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),  # upsample by 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # upsample by 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # upsample by 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # upsample by 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=1)  # final output layer
        )
    
    def forward(self, x):
        # Backbone (ResNet)
        x = self.backbone(x)
        
        # Decoder (upsampling to original image size)
        x = self.decoder(x)
        
        # Upsample to match the input image size (optional, if needed)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x

# Example usage:
model = ResNetSegmentation(num_classes=21, resnet_depth=50)
input_tensor = torch.randn(1, 3, 224, 224)  # Example input
output = model(input_tensor)

print(f"Output shape: {output.shape}")  # Should match the input resolution if scale_factor=2
