import os
import json

# Path to the folder containing images and JSON files
image_folder = './Manuscript'

# Initialize global counters
total_text = 0
total_equation = 0

# Iterate over all files in the folder
for file in os.listdir(image_folder):
    if file.endswith('.json'):  # Check if it is a JSON file
        json_path = os.path.join(image_folder, file)
        
        # Read the content of the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            annotations = data['shapes']  # Access annotations under the 'shapes' key

        # Initialize local counters for the current file
        text_count = 0
        equation_count = 0

        # Count the annotations
        for annotation in annotations:
            label_lower = annotation['label'].lower()
            if label_lower.startswith('text'):
                text_count += 1
            elif label_lower.startswith('equation'):
                equation_count += 1

        # Add the local counters to the global counters
        total_text += text_count
        total_equation += equation_count

        # Display the count for the current image
        print(f"{file.replace('.json', '')}: Text = {text_count}, Equation = {equation_count}")

# Display the total count for each type of annotation
print(f"Total text annotations: {total_text}")
print(f"Total equation annotations: {total_equation}")
