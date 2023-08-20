import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, MaxPooling2D, Add, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def mobile_hr_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Step 1: Input normalization of raw data
    x = inputs

    # Step 2: Function definition
    def conv_bn(x, filters, kernel_size=3, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Step 3: Kernel sizes and conv-batch norm
    x = conv_bn(x, filters=32)
    y = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(x)

    # Step 4: Depthwise Conv2D was used rather than Conv2D
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Step 5: Establishing the network with skip connections
    for _ in range(14):
        # Skip connection
        x = Add()([x, y])
        y = x

        # Depthwise Convolution
        x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Pointwise Convolution
        x = conv_bn(x, filters=32)

    # Step 6: Flattened layer and feature map extraction
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)

    # SVM Classifier
    svm_output = Dense(1, activation='linear')(x)

    # Create the Mobile-HR model
    model = Model(inputs=inputs, outputs=svm_output)

    return model

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have the Mobile-HR model and train_model_with_train_test_split functions defined

def train_model_with_train_test_split(X, y, input_shape, num_classes, test_size=0.2, batch_size=32, epochs=10):
    # Initialize arrays to store evaluation metrics for each fold
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    losses = []

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

    # Build the Mobile-HR model
    model = mobile_hr_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='hinge', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred >= 0).astype(int)

    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels)
    loss = history.history['loss'][-1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy)
    sensitivities.append(recall)
    specificities.append(specificity)
    f1_scores.append(f1)
    losses.append(loss)

    # Save the model weights
    model.save_weights(f'mobile_hr_model_weights.h5')

    return history, accuracies, sensitivities, specificities, f1_scores, losses
def train_model_with_k_fold(X, y, input_shape, num_classes, n_splits=5, batch_size=32, epochs=10):
    # Initialize arrays to store evaluation metrics for each fold
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    losses = []

    # Initialize lists to store training and validation histories for each fold
    training_histories = []
    validation_histories = []

    # Create a KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Loop through the folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training on Fold {fold + 1}...")
        
        # Get the training and testing data for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Build the Mobile-HR model
        model = mobile_hr_model(input_shape, num_classes)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='hinge', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

        # Evaluate the model on the test set for this fold
        y_pred = model.predict(X_test)
        y_pred_labels = (y_pred >= 0).astype(int)

        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels)
        recall = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)
        loss = history.history['loss'][-1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels).ravel()
        specificity = tn / (tn + fp)

        accuracies.append(accuracy)
        sensitivities.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        losses.append(loss)

        # Save the model weights for each fold
        model.save_weights(f'mobile_hr_model_weights_fold_{fold + 1}.h5')

        # Store the training and validation histories for this fold
        training_histories.append(history.history['accuracy'])
        validation_histories.append(history.history['val_accuracy'])

    # Calculate average accuracies and other metrics
    average_accuracy = np.mean(accuracies)
    average_sensitivity = np.mean(sensitivities)
    average_specificity = np.mean(specificities)
    average_f1_score = np.mean(f1_scores)

    # Plot training and validation curves for each fold
    plt.figure(figsize=(10, 6))
    for fold in range(n_splits):
        plt.plot(training_histories[fold], label=f'Training Fold {fold + 1}')
        plt.plot(validation_histories[fold], label=f'Validation Fold {fold + 1}')

    plt.title('Training and Validation Curves for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    return history, accuracies, sensitivities, specificities, f1_scores, losses, average_accuracy, average_sensitivity, average_specificity, average_f1_score
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


input_shape = (224, 224, 3)  # Adjust the input shape based on your image size
num_classes = 2  # Binary classification (normal or HR)

data_folder = 'C:\\Users\\arpit\\OneDrive\\Documents\\New Data\\Images'
image_filenames = os.listdir(data_folder)
num_images = len(image_filenames)
X = np.zeros((num_images, input_shape[0], input_shape[1], input_shape[2]))
y = np.zeros(num_images)

# Assuming the Excel file contains two columns: 'Image' and 'Label'
excel_file = 'C:\\Users\\arpit\\OneDrive\\Documents\\New Data\\ODIR groundtruth.xlsx'
df = pd.read_excel(excel_file)
for idx, filename in enumerate(image_filenames):
    img_path = os.path.join(data_folder, filename)
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    X[idx] = img_array

    # Assuming 'Image' column in the Excel file contains the image filenames and 'Label' column contains the labels
        
    if not df.empty and filename in df['Image'].values:
        label = df[df['Image'] == filename]['Hypertensive Retinopathy'].values[0]
        y[idx] = label
    else:
        print(f"Image {filename} not found in the DataFrame.")

# Convert labels to categorical format
y = np.expand_dims(y, axis=1)  # Add an extra dimension to make it (num_images, 1)
batch_size = 32
epochs = 50

 # Train the Mobile-HR model with train-test split
 # Train the Mobile-HR model with train-test split
history, accuracies, sensitivities, specificities, f1_scores, losses = train_model_with_train_test_split(
        X, y, input_shape, num_classes, batch_size=batch_size, epochs=epochs
)
# Plot training and validation loss/accuracy
plot_loss_and_accuracy(history)
print("Accuracy: ",accuracies, "\nSensitivity: ",sensitivities,"\nSpecificity: " ,specificities,"\nf1_score: ", f1_scores, "\nLoss: ",losses)
