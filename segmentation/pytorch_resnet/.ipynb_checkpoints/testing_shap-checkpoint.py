import torch
import shap
import numpy as np
from torchvision import models

# Load pre-trained DeepLabV3 model with ResNet backbone
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Generate random data (1 image, 3 channels, 224x224 resolution)
random_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
random_input_tensor = torch.tensor(random_input)

# Define the function that will be used for SHAP explanations
def model_forward(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)['out']
    return output

# Create SHAP explainer
explainer = shap.DeepExplainer(model_forward, random_input_tensor)

# Compute SHAP values
shap_values = explainer.shap_values(random_input_tensor)

# Visualize SHAP values
shap.image_plot(shap_values, random_input_tensor.numpy())
