import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensoflow.keras.optimizers import Adam

# Load the ResNet50 model without the top (classification) layers
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add your own classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification (hypertensive vs. normal)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Load the image filenames and labels from the Excel file
data = pd.read_excel('C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')
image_files = data['Image'].tolist()
labels = data['Hypertensive Retinopathy'].tolist()
test_data = pd.read_excel('C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')

data['Hypertensive Retinopathy'] = data['Hypertensive Retinopathy'].astype(str)
test_data['Hypertensive Retinopathy'] = test_data['Hypertensive Retinopathy'].astype(str)# Define the number of folds
k = 5

# Perform k-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store the evaluation results
accuracy_scores = []
loss_scores = []
overall_accuracy_history = []
overall_loss_history = []
overall_val_accuracy_history=[]
overall_val_loss_history=[]
history_list = []  # Modified: Initialize list to store training history

# Iterate over the folds
for fold, (train_index, valid_index) in enumerate(kf.split(data), 1):
    train_data = data.iloc[train_index]
    valid_data = data.iloc[valid_index]

    # Create data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        target_size=(224, 224)
    )

    valid_generator = train_datagen.flow_from_dataframe(
        dataframe=valid_data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        target_size=(224, 224)
    )

    history = model.fit(train_generator, validation_data=valid_generator, epochs=100, verbose=0)
    history_list.append(history)  # Modified: Append history to list

    # Save the weights for the current fold and epoch
    model.save_weights("Resnet50",f'fold{fold}_weights_epoch{fold}.h5')

    # Evaluate the model on the current fold
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/Test Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        batch_size=32,
        shuffle=False,
        class_mode='binary',
        target_size=(224, 224)
    )

    loss, accuracy = model.evaluate(test_generator)
    accuracy_scores.append(accuracy)
    loss_scores.append(loss)

    # Append accuracy and loss history for overall plot
    overall_accuracy_history.append(history.history['accuracy'])
    overall_loss_history.append(history.history['loss'])
    # Append validation accuracy and loss history for overall plot
    overall_val_accuracy_history.append(history.history['val_accuracy'])
    overall_val_loss_history.append(history.history['val_loss'])

# Calculate the mean accuracy and loss scores
mean_accuracy = np.mean(accuracy_scores)
mean_loss = np.mean(loss_scores)

print('Mean Test Accuracy:', mean_accuracy)
print('Mean Test Loss:', mean_loss)

for fold, history in enumerate(history_list, 1):  # Modified: Iterate over history_list
    plot_accuracy_fold(history)
    plot_loss_fold(history)

plot_accuracy_overall(overall_accuracy_history,overall_val_accuracy_history)
plot_loss_overall(overall_loss_history,overall_val_loss_history)

