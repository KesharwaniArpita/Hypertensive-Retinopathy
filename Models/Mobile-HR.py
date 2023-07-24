import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming one has the Mobile-HR model and train_model_with_train_test_split functions defined
# ... (code for the Mobile-HR model and train_model_with_train_test_split functions)

def plot_loss_and_accuracy(history):
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

if __name__ == "__main__":
    input_shape = (224, 224, 3)  # Adjust the input shape based on your image size
    num_classes = 2  # Binary classification (normal or HR)

    data_folder = 'c:\\Users\\DELL\\Documents\\New Data\\N'
    image_filenames = os.listdir(data_folder)
    num_images = len(image_filenames)

    X = np.zeros((num_images, input_shape[0], input_shape[1], input_shape[2]))
    y = np.zeros(num_images)

    # Assuming the Excel file contains two columns: 'Image' and 'Label'
    excel_file = 'c:\\Users\\DELL\\Documents\\New Data\\ODIR groundtruth.xlsx'
    df = pd.read_excel(excel_file)

    for idx, filename in enumerate(image_filenames):
        img_path = os.path.join(data_folder, filename)
        img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = img_to_array(img)
        X[idx] = img_array

        # Assuming 'Image' column in the Excel file contains the image filenames and 'Label' column contains the labels
        
        label = df[df['Image'] == filename]['H'].values[0]
        y[idx] = label

    # Convert labels to categorical format
    y = np.expand_dims(y, axis=1)  # Add an extra dimension to make it (num_images, 1)

    batch_size = 32
    epochs = 3

    # Train the Mobile-HR model with train-test split
    # Train the Mobile-HR model with train-test split
    history, accuracies, sensitivities, specificities, f1_scores, losses = train_model_with_train_test_split(
        X, y, input_shape, num_classes, batch_size=batch_size, epochs=epochs
    )
    # Plot training and validation loss/accuracy
    plot_loss_and_accuracy(history)
