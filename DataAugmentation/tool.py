import os
from PIL import Image
import json
import csv

def rename_images(folder):
    # List all files in the folder
    files = os.listdir(folder)

    # Filter to keep only images (common extensions)
    valid_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

    # Sort images alphabetically (optional)
    images.sort()

    # Rename each image
    for i, filename in enumerate(images, start=1):
        new_name = f"image{i}{os.path.splitext(filename)[1]}"
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
        print(f"'{filename}' renamed to '{new_name}'")

def split_image_vertically(image_path, output_folder):
    # Open the image
    img = Image.open(image_path)

    # Calculate the middle of the image
    width_middle = img.width // 2

    # Split the image in half
    g_img = img.crop((0, 0, width_middle, img.height))
    d_img = img.crop((width_middle, 0, img.width, img.height))

    # Create file names for the halves
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    g_file = os.path.join(output_folder, f"{base_name}_g.jpg")
    d_file = os.path.join(output_folder, f"{base_name}_d.jpg")

    # Save both halves
    g_img.save(g_file)
    d_img.save(d_file)

def split_image_vertically_folder(source_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the folder
    for file in os.listdir(source_folder):
        full_path = os.path.join(source_folder, file)

        # Check if it is a file and an image
        if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            split_image_vertically(full_path, output_folder)


def remove_bottom_image(image_path, pixels_to_remove, output_folder):
    with Image.open(image_path) as img:
        # Calculate the new dimensions
        width, height = img.size
        new_height = height - pixels_to_remove

        # Crop the image
        cropped_img = img.crop((0, 0, width, new_height))

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the modified image
        filename = os.path.basename(image_path)
        cropped_img.save(os.path.join(output_folder, filename))

def remove_Bottom_image_folder(source_folder, output_folder, pixels_to_remove):
    for file in os.listdir(source_folder):
        full_path = os.path.join(source_folder, file)
        if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            remove_bottom_image(full_path, pixels_to_remove, output_folder)

# Example of usage for cropping
pixels_to_remove = 315
remove_Bottom_image_folder('./OriginalImage', './DataAugmentation/Input', pixels_to_remove)
# Example of usage for splitting
split_image_vertically_folder('./DataAugmentation/Input', './DataAugmentation/Input')

