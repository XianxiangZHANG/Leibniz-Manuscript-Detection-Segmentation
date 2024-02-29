import os
import json
import cv2
import numpy as np

def is_pixel_black(pixel, threshold=100):
    return pixel < threshold

def point_in_polygon(x, y, polygon):
    polygon_np = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(polygon_np, (x, y), False) >= 0

def analyze_images(image_folder):
    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            base_name, _ = os.path.splitext(filename)
            json_path = os.path.join(image_folder, f"{base_name}.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                annotations = json.load(f)

            text_pixels = 0
            equation_pixels = 0
            background_pixels = 0

            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    pixel = image[y, x]
                    is_text = is_equation = False

                    for shape in annotations['shapes']:
                        if point_in_polygon(x, y, shape['points']):
                            if 'text' in shape['label'].lower() and is_pixel_black(pixel):
                                text_pixels += 1
                                is_text = True
                            elif 'equation' in shape['label'].lower() and is_pixel_black(pixel):
                                equation_pixels += 1
                                is_equation = True

                    if not is_text and not is_equation:
                        background_pixels += 1 if is_pixel_black(pixel, threshold=255) else 0

            total_pixels = image.shape[0] * image.shape[1]
            results.append({
                'image': filename,
                'text_proportion': text_pixels / total_pixels,
                'equation_proportion': equation_pixels / total_pixels,
                'background_proportion': background_pixels / total_pixels,
                'text_proportion_excl_background': text_pixels / (text_pixels + equation_pixels),
                'equation_proportion_excl_background': equation_pixels / (text_pixels + equation_pixels),
            })

    averages = {
        'average_text': np.mean([r['text_proportion'] for r in results]),
        'average_equation': np.mean([r['equation_proportion'] for r in results]),
        'average_background': np.mean([r['background_proportion'] for r in results]),
        'average_text_excl_background': np.mean([r['text_proportion_excl_background'] for r in results]),
        'average_equation_excl_background': np.mean([r['equation_proportion_excl_background'] for r in results])
    }

    return results, averages

# Folder containing images and JSON files for analysis
image_folder = './Input'
results, averages = analyze_images(image_folder)

for result in results:
    print(result)

print("Averages for all images:")
print(averages)
