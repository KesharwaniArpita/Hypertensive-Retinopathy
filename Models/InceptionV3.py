import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np



def train_and_plot_model(num_epochs):
    # Load the data from the Excel file
    data = pd.read_excel('C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')
    data['Hypertensive Retinopathy'] = data['Hypertensive Retinopathy'].astype(str)

    # Split the data into training and validation sets
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # Load the InceptionV3 model without the top (fully connected) layers
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the pre-trained layers so they are not trained during the fine-tuning process
    for layer in inception_model.layers:
        layer.trainable = False

    # Add custom fully connected layers on top of the InceptionV3 base model
    model = Sequential()
    model.add(inception_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, validation_data=valid_generator, epochs=num_epochs, verbose=0)

    # Evaluate the model on the test set
    test_generator = valid_datagen.flow_from_dataframe(
        dataframe=data,
        directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
        x_col='Image',
        y_col='Hypertensive Retinopathy',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    loss, accuracy = model.evaluate(test_generator)
    print('Test Accuracy:', accuracy)

    # Plot the epoch-accuracy graph
    plot_accuracy(history,"InceptionV3")

# Call the function to train and plot the model for 500 epochs
train_and_plot_model(num_epochs=1)
