import unittest
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        # Specify the directory where the fundus images are stored
        self.image_dir = 'C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set'
        self.csv_file = 'C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv'

        # Specify the target image size
        self.target_size = (224, 224)

    def test_random_forest(self):
        # Load the labels from the CSV file using Pandas
        data = pd.read_csv(self.csv_file)

        # Load the pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.target_size[0], self.target_size[1], 3))

        # Remove the classification layers from the model
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

        # Initialize empty lists for features and labels
        features = []
        labels = []

        # Load the fundus images and extract features
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.target_size)
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

        # Define the parameter grid for Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize a Random Forest classifier
        classifier = RandomForestClassifier(random_state=280)

        # Perform Grid Search to find the best hyperparameters
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3)
        grid_search.fit(train_features, train_labels)

        # Get the best estimator and its hyperparameters
        best_classifier = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Train the model on the training data with the best hyperparameters
        best_classifier.fit(train_features, train_labels)

        # Use the trained model to predict labels for the test features
        predictions = best_classifier.predict(test_features)

        # Evaluate the model's performance
        accuracy = accuracy_score(test_labels, predictions)
        confusion_mtx = confusion_matrix(test_labels, predictions)

        # Assert that the accuracy is within a reasonable range (0.0 to 1.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

        # Assert that the confusion matrix is a 2x2 matrix
        self.assertEqual(confusion_mtx.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
