import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

def preprocess_image(image_path, output_folder, df):
    # Load the image
    img = Image.open(image_path)

    # Step 1: Set missing or incorrect values of pixels and remove outliers (if needed)
    # Replace missing or incorrect values with a specified value (e.g., 0)
    img_array = np.array(img)
    img_array[img_array < 0] = 0  # Replace negative values with 0
    img_array[img_array > 255] = 0  # Replace values greater than 255 with 0

    # Step 2: Feature engineering - normalization of variables (Feature Scaling)
    img_array = np.array(img)
    img_array_scaled = img_array / 255.0  # Scale pixel values to the range [0, 1]
    scaled_img = Image.fromarray((img_array_scaled * 255).astype(np.uint8))

    # Save the normalized image with the "scaled_" prefix added to the original image name
    img_name = os.path.basename(image_path)
    img_name_no_ext, img_ext = os.path.splitext(img_name)
    scaled_img_path = os.path.join(output_folder, f"scaled_{img_name}")  # Add "scaled_" prefix
    scaled_img.save(scaled_img_path)

    # Assuming the Excel file contains two columns: 'Image' and 'Label'
    label = df[df['Image'] == img_name]['H'].values[0]

    # Update the DataFrame with the image name (with "scaled_" prefix) and label
    df = df.append({'Image': f"scaled_{img_name}", 'H': label}, ignore_index=True)

    return df

if __name__ == "__main__":
    image_folder = 'C:\\Users\\DELL\\Documents\\New Data\\Images'
    output_folder = 'C:\\Users\\DELL\\Documents\\New Data\\Preprocessed images'
    excel_file = 'C:\\Users\\DELL\\Documents\\New Data\\ODIR groundtruth.xlsx'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the Excel file
    df = pd.read_excel(excel_file)

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            df = preprocess_image(image_path, output_folder, df)

    # Save the updated DataFrame back to the Excel file
    df.to_excel(excel_file, index=False)
