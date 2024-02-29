

# README for Data Augmentation


This toolkit comprises four scripts designed to facilitate the analysis and processing of image annotations. Each script serves a specific purpose within the context of image annotation management and analysis.

## 1. countAnnotation.py

**Purpose:** This script counts annotations within JSON files, differentiating between text and equation annotations.

**How to Use:**

- Place your JSON files in a specified directory.
- Update the `image_folder` variable in the script to point to your directory.
- Execute the script using the command below:

```bash
python countAnnotation.py
```

The script outputs the count of 'text' and 'equation' annotations for each JSON file and provides a total count across all processed files.

## 2. dataAugmentationJson.py

**Purpose:** This script is designed to augment image annotation data, allowing you to generate new annotated images by placing text and equation elements on different backgrounds without overlap. The script selects a specified number of text and equation annotations from the input data and arranges them on a new background, generating both the augmented images and their corresponding JSON annotation files.

**How to Use:**

1. **Prepare Your Data:**
   - Organize your original images and JSON files in a designated input folder.
   - Ensure you have a background image to use for placing the annotations.

2. **Configure the Script:**
   - Set the `folder`, `output_folder`, `background_path`, `num_images`, `num_equations`, `num_texts`, `width`, and `height` variables in the script to match your specific requirements.

3. **Execute the Script:**
   - Run the following command in your terminal:
   ```bash
   python dataAugmentationJson.py
   ```
   - This command will process the specified number of images, extracting and repositioning text and equation annotations onto the designated background images, and then save the augmented images and their new annotations into the `output_folder`.

4. **Review the Output:**
   - Check the `output_folder` for the generated images and their corresponding JSON files.
   - Optionally, you can also create CSV files from the JSON annotations for further analysis or processing by calling the `create_csvImage` function within the script.

By following these steps, you can effectively use `dataAugmentationJson.py` to enhance your dataset for machine learning or other analytical purposes, leveraging existing annotations in a new visual context.

## 3. pixelMetricsAnnotation.py

**Purpose:** This script analyzes pixel-level metrics within annotated image areas, distinguishing between text, equation, and background pixels based on their annotations. It calculates the proportion of each type within individual images and averages these proportions across a dataset.
This code takes very long time to run

**How to Use:**

1. **Prepare Your Dataset:**
   - Ensure that your image folder contains both the images and their corresponding JSON annotation files.

2. **Configuration:**
   - Set the `image_folder` variable to the directory containing your image and JSON files.

3. **Execution:**
   - Run the script with the following command:
   ```bash
   python pixelMetricsAnnotation.py
   ```
   - The script processes each image and its annotations to determine the proportions of text, equation, and background pixels, outputting these metrics for individual images and computing averages across all analyzed images.

4. **Reviewing Results:**
   - The script prints the calculated proportions for each image and the overall averages to the console.
   - Use this data for further analysis, reporting, or to guide data augmentation and machine learning training processes.

This script offers valuable insights into the composition of annotated image datasets, aiding in the understanding of data distribution and potential biases.

## 4. tool.py

**Purpose:** This script provides utilities for processing images, including renaming images for consistency, splitting images vertically into two halves, and removing a specified pixel amount from the bottom of images. It's designed to aid in batch processing of image files for organizational or preprocessing steps before further analysis or data augmentation.

**How to Use:**

1. **Rename Images:**
   - Organize your images into a folder.
   - Execute `rename_images(folder)` to rename all images in a sequential order.

2. **Split Images Vertically:**
   - Use `split_image_vertically(image_path, output_folder)` to divide an image into left and right halves, saving them separately.

3. **Remove Bottom from Images:**
   - Apply `remove_bottom_image(image_path, pixels_to_remove, output_folder)` to crop out a specified pixel amount from the image's bottom.

4. **Batch Processing:**
   - For folder-based operations, use `split_image_vertically_folder(source_folder, output_folder)` and `remove_Bottom_image_folder(source_folder, output_folder, pixels_to_remove)` to process multiple images.

**Example Commands:**

- To rename images within a folder:
  ```bash
  python tool.py rename_images ./path/to/image/folder
  ```

- To split images in a folder:
  ```bash
  python tool.py split_image_vertically_folder ./path/to/source/folder ./path/to/output/folder
  ```

- To remove the bottom portion of images:
  ```bash
  python tool.py remove_Bottom_image_folder ./path/to/source/folder ./path/to/output/folder 315
  ```

**Note:** The actual command-line interface may vary; adjust parameters and function calls based on your specific setup and the script's content.


**General Execution Steps:**

- Ensure Python is installed on your machine.
- Save the scripts in a convenient location.
- Adjust any file paths or parameters within the scripts as necessary.
- Run each script from the command line within its directory or specify the path to the script.

**Example Command:**

```bash
python <script_name>.py
```

Replace `<script_name>` with the actual name of the script you wish to run.



---
