import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Set the paths for your image folder, Excel file, and the output folder for augmented images
image_folder = 'C:\\Users\\DELL\\Documents\\New Data\\Images'
excel_file_path = 'C:\\Users\\DELL\\Documents\\New Data\\ODIR groundtruth.xlsx'
output_folder = 'C:\\Users\\DELL\\Documents\\New Data\\Augmented_Images'

# Read the Excel file and create a DataFrame
data = pd.read_excel(excel_file_path)

# Define the ImageDataGenerator with the desired augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through the DataFrame and perform data augmentation for the minority class
for index, row in data.iterrows():
    if row['Hypertensive Retinopathy'] == 1:  # Replace 'minority_label' with the numerical value of the minority class label
        img_path = os.path.join(image_folder, row['Image'])  # Replace 'Image_Name_Column' with the column name containing image names
        # Load the image and resize it to the desired target size (e.g., 224x224)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add the batch size dimension

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Perform data augmentation for the minority class images
        augmented_count = 0
        for batch in datagen.flow(
            img,
            batch_size=1,
            save_to_dir=output_folder,
            save_prefix=f"augmented_{index}",
            save_format='jpg'
        ):
            augmented_count += 1
            if augmented_count >= 8:  # Augment the minority class to 10 times
                break

            # Append the names of augmented images to the DataFrame
            augmented_image_name = f"augmented_{index}_{augmented_count:04d}.jpg"
            data = data.append({'Image': augmented_image_name, 'H': 1}, ignore_index=True)

# Save the updated DataFrame back to the Excel file
data.to_excel(excel_file_path, index=False)

