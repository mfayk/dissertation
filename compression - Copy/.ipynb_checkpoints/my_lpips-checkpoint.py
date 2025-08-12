import torch
import lpips
from torchvision import transforms
from PIL import Image
import numpy as np


# Load LPIPS model
lpips_model = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze' instead of 'alex'

# Load the images
image1 = Image.open("inps_array_img.png").convert("RGB")  # Replace with your image file path
image2 = Image.open("decomp_array_img.png").convert("RGB")  # Replace with your image file path
# Check if the images are the same (quick sanity check)
image1_array = np.array(image1)
image2_array = np.array(image2)

if np.array_equal(image1_array, image2_array):
    print("Images are identical!")
else:
    print("Images are different.")

# Transform images to tensor and normalize (resize if necessary)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image1_tensor = transform(image1).unsqueeze(0)  # Add batch dimension
image2_tensor = transform(image2).unsqueeze(0)  # Add batch dimension

# Calculate LPIPS distance
with torch.no_grad():
    dist = lpips_model(image1_tensor, image2_tensor)

# Output the distance score
print(f"LPIPS distance score: {dist.item()}")