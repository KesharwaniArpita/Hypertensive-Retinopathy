import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Specify the directory where the fundus images are stored
# Specify the directory where the fundus images are stored
image_dir = 'C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/Overall'
excel_file = 'C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx'

# Specify the target image size
target_size = (512, 512)

# Load the labels from the Excel file using Pandas
data = pd.read_excel(excel_file)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

# Remove the classification layers from the model
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Initialize empty lists for features and labels
features = []
labels = []

# Load the fundus images and extract features
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        
        # Extract features from the image
        features.append(feature_extractor.predict(np.expand_dims(img, axis=0))[0])
        
        # Get the label corresponding to the image filename from the CSV file
        label = data.loc[data['Image'] == filename]['Hypertensive Retinopathy'].values[0]
        labels.append(label)

# Convert the features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Reshape the features to a 2-dimensional array
num_samples, height, width, num_channels = features.shape
features = features.reshape(num_samples, height * width * num_channels)

# Split the data into training and testing subsets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
classifier.fit(train_features, train_labels)

# Use the trained model to predict labels for the test features
predictions = classifier.predict(test_features)
# Calculate the Kappa score
kappa = cohen_kappa_score(test_labels, predictions)

# Calculate the F1 score
f1 = f1_score(test_labels, predictions, average='weighted')

# Calculate the specificity
tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
specificity = tn / (tn + fp)

# Print the Kappa score, F1 score, and specificity
print('Kappa Score:', kappa)
print('F1 Score:', f1)
print('Specificity:', specificity)
# Evaluate the model's performance
accuracy = accuracy_score(test_labels, predictions)
confusion_mtx = confusion_matrix(test_labels, predictions)

# Print the accuracy and confusion matrix
print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_mtx)
