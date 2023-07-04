import unittest
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class ResNet50Model(unittest.TestCase):
    def setUp(self):
        # Load the data
        self.data = pd.read_excel('path/to/excel/file.xlsx')

        # Create the ResNet50 model
        base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def ResNet_50_model(self):
        # Define the number of folds
        k = 5

        # Perform k-fold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # Initialize lists to store the evaluation results
        accuracy_scores = []
        loss_scores = []

        # Iterate over the folds
        for fold, (train_index, valid_index) in enumerate(kf.split(self.data), 1):
            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

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

            self.model.fit(train_generator, validation_data=valid_generator, epochs=1)
            
            # Save the weights for the current fold
            self.model.save_weights(f'fold{fold}_weights.h5')

            # Evaluate the model on the current fold
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_dataframe(
                dataframe=valid_data,
                directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
                x_col='Image',
                y_col='Hypertensive Retinopathy',
                batch_size=32,
                shuffle=False,
                class_mode='binary',
                target_size=(224, 224)
            )
            loss, accuracy = self.model.evaluate(test_generator)
            accuracy_scores.append(accuracy)
            loss_scores.append(loss)

        # Calculate the mean accuracy and loss scores
        mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        mean_loss = sum(loss_scores) / len(loss_scores)

        # Assert statements to check the test results
        self.assertGreater(mean_accuracy, 0.0)
        self.assertLess(mean_loss, 1.0)

if __name__ == '__main__':
    unittest.main()
