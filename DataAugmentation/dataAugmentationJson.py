import csv
import cv2
import json
import numpy as np
import random
from PIL import Image
import os


# Create a binary mask from a set of polygon points.
def create_mask_from_points(points, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

# Read image and JSON files from a specified folder.
def read_files_in_folder(folder):
    files = os.listdir(folder)
    images = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    jsons = [f for f in files if f.endswith('.json')]
    return images, jsons

# Collect and pair all image and annotation data from the given folder.
def collect_all_elements(jsons, folder):
    all_elements = []
    for json_name in jsons:
        with open(os.path.join(folder, json_name), 'r') as f:
            annotations = json.load(f)
        for annotation in annotations['shapes']:
            element = (os.path.join(folder, json_name.replace('.json', '.jpg')), annotation)
            all_elements.append(element)
    return all_elements

# Select a specific number of text and equation elements from the collected elements.
def select_elements(all_elements, num_equations, num_texts):
    random.shuffle(all_elements)
    equations_elements = []
    texts_elements = []
    for img_path, annotation in all_elements:
        label = annotation['label'].lower()
        if 'text' in label and len(texts_elements) < num_texts:
            texts_elements.append((img_path, annotation))
        elif 'equation' in label and len(equations_elements) < num_equations:
            equations_elements.append((img_path, annotation))
        if len(texts_elements) == num_texts and len(equations_elements) == num_equations:
            break
    return texts_elements + equations_elements

# Resize an image to the specified width and height.
def resize_image(image_path, width, height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Place elements on a background without overlapping, respecting the given image size.
def place_elements_without_collision(background, elements, output_folder, i, image_size, image_path):
    annotations_resulting = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": f'image_{i}.jpg',
        "imageData": None,
        "imageHeight": image_size[1],
        "imageWidth": image_size[0]
    }
    occupied_zones = []

    for img_path, element in elements:
        image = cv2.imread(img_path)
        points = np.array(element['points'], dtype=np.int32)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect

        if w >= image_size[0] or h >= image_size[1]:
            print(f"Skipping element {element['label']} due to size.")
            continue

        cropped_element = image[y:y+h, x:x+w]
        mask = create_mask_from_points(points - [x, y], cropped_element.shape)

        collision = True
        attempts = 0
        while collision and attempts < 100:
            new_x = np.random.randint(0, image_size[0] - w)
            new_y = np.random.randint(0, image_size[1] - h)
            collision = any(new_x < x2 + w2 and new_x + w > x2 and new_y < y2 + h2 and new_y + h > y2 for x2, y2, w2, h2 in occupied_zones)
            attempts += 1

        if not collision:
            background[new_y:new_y+h, new_x:new_x+w][mask > 0] = cropped_element[mask > 0]
            new_points = [[int(point[0] - x + new_x), int(point[1] - y + new_y)] for point in points]

            annotations_resulting['shapes'].append({
                "label": element['label'],
                "points": new_points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
            occupied_zones.append((new_x, new_y, w, h))

    result_image_path = os.path.join(output_folder, f'image_{i}.jpg')
    cv2.imwrite(result_image_path, background)
    annotations_resulting['imagePath'] = os.path.basename(result_image_path)

    with open(os.path.join(output_folder, f'image_{i}.json'), 'w') as f:
        json.dump(annotations_resulting, f, indent=4)

# Generate images with annotations, placing text and equations without overlaps.
def generate_images_with_annotations(folder, output_folder, background_path, num_images, num_equations, num_texts, width, height):
    _, jsons = read_files_in_folder(folder)
    all_elements = collect_all_elements(jsons, folder)
    background = resize_image(background_path, width, height)
    image_size = (width, height)

    for i in range(num_images):
        selected_elements = select_elements(all_elements, num_equations, num_texts)
        place_elements_without_collision(background.copy(), selected_elements, output_folder, i, image_size, background_path)



def create_csvImage(folder):
    for file in os.listdir(folder):
        if file.endswith('.json'):
            path_json = os.path.join(folder, file)
            path_image = os.path.splitext(path_json)[0] + '.jpg'

            # Charger le file JSON
            with open(path_json, 'r') as jsonfile:
                data = json.load(jsonfile)
                annotations = data['shapes']  # Récupérer toutes les annotations dans le file JSON

                # Créer un file CSV pour chaque image
                file_csv = os.path.splitext(path_json)[0] + '.csv'

                # Ouvrir le file CSV en mode écriture
                with open(file_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Nom de l\'image', 'Label', 'Coordonnées des points'])  # En-têtes du CSV

                    # Écrire toutes les annotations dans le file CSV
                    for annotation in annotations:
                        label_name = annotation['label']  # Récupérer le nom du label
                        points = annotation['points']  # Récupérer les coordonnées des points

                        # Écrire les données dans le file CSV
                        writer.writerow([path_image, label_name, points])
                        
# Example of how to use the function with specified image dimensions.


num_images = 2
num_equations = 20
num_texts = 20
folder = './Input'
output_folder = './Output'
background_path = './Background/fond.jpg'
width = 1024  # Width of the output images
height = 1024  # Height of the output images
generate_images_with_annotations(folder, output_folder, background_path, num_images, num_equations, num_texts, width, height)


# Assumes createCsv.create_csvImage function exists and properly configured to create a CSV of image data.
create_csvImage('./output_folder')
