import lpips
import torch
from PIL import Image
from torchvision import transforms

# Load pre-trained LPIPS model (e.g., AlexNet-based)
lpips_model = lpips.LPIPS(net='alex')

# Load and preprocess images
def load_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2.0 - 1.0)  # Normalize to [-1, 1]
    ])
    return transform(img).unsqueeze(0)

img1 = load_image("inps_array_img.png")
img2 = load_image("decomp_array_img.png")

# Compute LPIPS score
dist_score = lpips_model(img1, img2)
print("LPIPS:", dist_score.item())