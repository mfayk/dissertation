#python file to convert


from PIL import Image
import os

# Define the folder path
input_folder = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split/train/rgb'
output_folder = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split-png/train/rgb'  # or set to another path if you want to save elsewhere

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
        jpeg_path = os.path.join(input_folder, filename)
        png_filename = os.path.splitext(filename)[0] + '.png'
        png_path = os.path.join(output_folder, png_filename)

        # Open and convert image
        with Image.open(jpeg_path) as img:
            img.convert('RGB').save(png_path, 'PNG')  # RGB ensures no alpha issues

        print(f'Converted {filename} to {png_filename}')


        
        # Define the folder path
input_folder = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split/test/rgb'
output_folder = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split-png/test/rgb'  # or set to another path if you want to save elsewhere

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
        jpeg_path = os.path.join(input_folder, filename)
        png_filename = os.path.splitext(filename)[0] + '.png'
        png_path = os.path.join(output_folder, png_filename)

        # Open and convert image
        with Image.open(jpeg_path) as img:
            img.convert('RGB').save(png_path, 'PNG')  # RGB ensures no alpha issues

        print(f'Converted {filename} to {png_filename}')

