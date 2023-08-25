import unittest
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from your_model_file import mobile_hr_model, train_model_with_train_test_split, train_model_with_k_fold, plot_loss_and_accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class TestMobileHRModel(unittest.TestCase):
    def setUp(self):
        # Set up necessary variables, paths, and data here
        self.input_shape = (224, 224, 3)
        self.num_classes = 2
        self.data_folder = 'path/to/data/folder'
        self.excel_file = 'path/to/excel/file.xlsx'
        
        # Load and preprocess data
        self.image_filenames = os.listdir(self.data_folder)
        self.num_images = len(self.image_filenames)
        self.X = np.zeros((self.num_images, *self.input_shape))
        self.y = np.zeros(self.num_images)
        
        df = pd.read_excel(self.excel_file)
        for idx, filename in enumerate(self.image_filenames):
            img_path = os.path.join(self.data_folder, filename)
            img = load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
            img_array = img_to_array(img)
            self.X[idx] = img_array

            if not df.empty and filename in df['Image'].values:
                label = df[df['Image'] == filename]['Hypertensive Retinopathy'].values[0]
                self.y[idx] = label

    def test_mobile_hr_model(self):
        model = mobile_hr_model(self.input_shape, self.num_classes)
        self.assertIsNotNone(model)

    def test_train_model_with_train_test_split(self):
        history, accuracies, sensitivities, specificities, f1_scores, losses = train_model_with_train_test_split(
            self.X, self.y, self.input_shape, self.num_classes, batch_size=32, epochs=5
        )
        self.assertIsNotNone(history)
        self.assertIsNotNone(accuracies)
        self.assertIsNotNone(sensitivities)
        self.assertIsNotNone(specificities)
        self.assertIsNotNone(f1_scores)
        self.assertIsNotNone(losses)

    def test_train_model_with_k_fold(self):
        history, accuracies, sensitivities, specificities, f1_scores, losses, average_accuracy, average_sensitivity, average_specificity, average_f1_score = train_model_with_k_fold(
            self.X, self.y, self.input_shape, self.num_classes, n_splits=5, batch_size=32, epochs=5
        )
        self.assertIsNotNone(history)
        self.assertIsNotNone(accuracies)
        self.assertIsNotNone(sensitivities)
        self.assertIsNotNone(specificities)
        self.assertIsNotNone(f1_scores)
        self.assertIsNotNone(losses)
        self.assertIsNotNone(average_accuracy)
        self.assertIsNotNone(average_sensitivity)
        self.assertIsNotNone(average_specificity)
        self.assertIsNotNone(average_f1_score)

    def test_plot_loss_and_accuracy(self):
        # Create a dummy history object for testing the plot function
        dummy_history = {'loss': [0.1, 0.2, 0.3], 'val_loss': [0.2, 0.3, 0.4],
                         'accuracy': [0.9, 0.85, 0.88], 'val_accuracy': [0.88, 0.87, 0.89]}
        plot_loss_and_accuracy(dummy_history)

if __name__ == '__main__':
    unittest.main()

