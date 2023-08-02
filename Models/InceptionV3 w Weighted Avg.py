import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess your image dataset
excel_file = '/content/drive/MyDrive/Balanced data groundtruth.xlsx'
df = pd.read_excel(excel_file)
#df['Hypertensive Retinopathy'] = df['Hypertensive Retinopathy'].astype(str)
# Step 2: Assign binary labels (0 for HR-ve and 1 for HR+ve) based on the weighted average
label_weights = {
    0: 1.0,  # Weight for HR-ve (Hypertensive Retinopathy negative)
    1: 2.0,  # Weight for HR+ve (Hypertensive Retinopathy positive)
}
df['Binary_Label'] = df['Hypertensive Retinopathy'].map(label_weights).astype(str)

# Step 3: Split your dataset into training, validation, and testing sets
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Create data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255
)

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/content/drive/MyDrive/Balanced Data',
    x_col='Image',
    y_col='Binary_Label',
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    target_size=(224, 224),
    class_weight={0: label_weights[0], 1: label_weights[1]}  # Set class_weight for training generator
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='/content/drive/MyDrive/Balanced Data',
    x_col='Image',
    y_col='Binary_Label',
    batch_size=32,
    shuffle=False,
    class_mode='binary',
    target_size=(224, 224)
)

# Assuming the Excel file for test data contains two columns: 'Image' and 'Binary_Label'
test_excel_file = '/content/drive/MyDrive/Balanced data groundtruth.xlsx'
test_df = pd.read_excel(test_excel_file)
test_df['Binary_Label'] = test_df['Hypertensive Retinopathy'].map(label_weights).astype(str)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/content/drive/MyDrive/Test',  # Modify this path accordingly
    x_col='Image',
    y_col='Binary_Label',
    batch_size=32,
    shuffle=False,
    class_mode='binary',
    target_size=(224, 224)
)
# Step 5: Build and train your model
# Load the InceptionV3 model without the top layers
base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add your own classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification (HR+ve or HR-ve)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with class_weight parameter
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation data
history = model.fit(train_generator, validation_data=valid_generator, epochs=200)

# Evaluate the model on the test data
y_true = test_generator.labels
y_pred = model.predict(test_generator)
y_pred_labels = (y_pred >= 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred_labels)
f1 = f1_score(y_true, y_pred_labels)

print('Test Accuracy:', accuracy)
print('Test F1 Score:', f1)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

