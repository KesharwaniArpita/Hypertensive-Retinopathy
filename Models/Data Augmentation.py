#DATA Augmentation
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def add_noise(img, noise_level=0.1):
    row, col, ch = img.shape
    mean = 0
    sigma = noise_level * 255.0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = img + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def adjust_contrast(img, alpha=1.5):
    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return contrast_img

def perform_data_augmentation(input_folder, output_folder, num_augmented_images):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.png')]

    for input_file in input_files:
        img = tf.keras.preprocessing.image.load_img(input_file)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix='augmented', save_format='png'):
            batch = batch[0]
            noisy_img = add_noise(batch, noise_level=0.1)
            contrast_img = adjust_contrast(batch, alpha=1.5)

            augmented_img = np.vstack([batch, noisy_img, contrast_img])
            cv2.imwrite(os.path.join(output_folder, f"augmented_{i}_{os.path.basename(input_file)}"), augmented_img)
            
            i += 1
            if i >= num_augmented_images:
                break
