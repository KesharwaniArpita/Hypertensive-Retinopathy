from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


# Load the DenseNet121 model without the top (classification) layers
base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))

# binary classification (hypertensive vs. normal)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the image filenames and labels from the Excel file
data = pd.read_excel('/content/drive/MyDrive/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')
image_files = data['Image'].tolist()
labels = data['Hypertensive Retinopathy'].tolist()

data['Hypertensive Retinopathy'] = data['Hypertensive Retinopathy'].astype(str)

# Create data generator for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory='/content/drive/MyDrive/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
    x_col='Image',
    y_col='Hypertensive Retinopathy',
    subset='training',
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    target_size=(224, 224)
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory='/content/drive/MyDrive/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
    x_col='Image',
    y_col='Hypertensive Retinopathy',
    subset='validation',
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    target_size=(224, 224)
)

# Train the model
history = model.fit(train_generator, validation_data=valid_generator, epochs=10, verbose=0)

# Evaluate the model on the test set
test_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory='/content/drive/MyDrive/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
    x_col='Image',
    y_col='Hypertensive Retinopathy',
    batch_size=32,
    shuffle=False,
    class_mode='binary',
    target_size=(224, 224)
)

loss, accuracy = model.evaluate(test_generator)
print('Test Accuracy:', accuracy)
plot_accuracy(history, "DenseNet121")
