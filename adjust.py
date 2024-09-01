import os
import random
from PIL import Image, ImageEnhance
import shutil

# Define paths
TEST_FOLDER = "./Images/Test"
NEW_TRAIN_FOLDER = "./Images/NewTrain"

# Create NewTrain folder if it doesn't exist
if not os.path.exists(NEW_TRAIN_FOLDER):
    os.makedirs(NEW_TRAIN_FOLDER)

# Number of images to process
N = 20000  # You can adjust this number

# Get list of all images in the test folder
all_images = [f for f in os.listdir(TEST_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Randomly select N images
selected_images = random.sample(all_images, min(N, len(all_images)))

for image_name in selected_images:
    # Open the image
    img_path = os.path.join(TEST_FOLDER, image_name)
    img = Image.open(img_path)

    # Convert the image to grayscale
    adjusted_img = img.convert('L')

    # Randomly adjust brightness
    brightness_enhancer = ImageEnhance.Brightness(adjusted_img)
    brightness_factor = random.uniform(0.2, 2)
    adjusted_img = brightness_enhancer.enhance(brightness_factor)

    # Randomly adjust contrast
    contrast_enhancer = ImageEnhance.Contrast(adjusted_img)
    contrast_factor = random.uniform(0.2, 3)
    adjusted_img = contrast_enhancer.enhance(contrast_factor)

    # Randomly adjust position by a random amount between +-10 pixels in each direction
    width, height = adjusted_img.size
    left = random.randint(-10, 10)
    top = random.randint(-10, 10)
    right = width + random.randint(-10, 10)
    bottom = height + random.randint(-10, 10)
    adjusted_img = adjusted_img.crop((left, top, right, bottom))

    # Resize back to original dimensions if necessary
    if adjusted_img.size != (width, height):
        adjusted_img = adjusted_img.resize((width, height), Image.LANCZOS)

    # Save the adjusted image to NewTrain folder
    new_img_name = f"{image_name}"
    new_img_path = os.path.join(NEW_TRAIN_FOLDER, new_img_name)
    adjusted_img.save(new_img_path)

print(f"Processed {len(selected_images)} images. Adjusted images saved in {NEW_TRAIN_FOLDER}")
