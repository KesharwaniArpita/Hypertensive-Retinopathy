import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Load the InceptionV3 model without the top (classification) layers
base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add your own classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification (hypertensive vs. normal)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Load the image filenames and labels from the Excel file
data = pd.read_excel('C:\\Users\\DELL\\Documents\\New Data\\ODIR groundtruth.xlsx', engine='openpyxl')
image_files = data['Image'].tolist()
labels = data['H'].tolist()

test_data = pd.read_excel('C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')

data['H'] = data['H'].astype(str)
test_data['Hypertensive Retinopathy'] = test_data['Hypertensive Retinopathy'].astype(str)

# Define the number of folds
k = 2

# Perform k-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store the evaluation results
accuracy_scores = []
loss_scores = []
overall_accuracy_history = []
overall_loss_history = []
overall_val_accuracy_history=[]
overall_val_loss_history=[]
history_list = []  # Initialize list to store training history
sensitivities = []
specificities = []
f1_scores = []



    # Split the data into training and testing sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the data into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='C:\\Users\\DELL\\Documents\\New Data\\AUG_IM',
    x_col='Image',
    y_col='H',
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    target_size=(224, 224)
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=valid_data,
    directory='C:\\Users\\DELL\\Documents\\New Data\\AUG_IM',
    x_col='Image',
    y_col='H',
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    target_size=(224, 224)
)

# Train the model with validation data
history = model.fit(train_generator, validation_data=valid_generator, epochs=200, verbose=1)

history_list.append(history)  # Append history to list

    # Save the weights for the current fold and epoch
#model.save_weights(f'fold{fold}_weights_epoch{fold}.h5')

    # Evaluate the model on the current fold
# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/Overall',
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

# Calculate additional evaluation metrics
y_pred = model.predict(test_generator)
y_pred_labels = (y_pred >= 0.5).astype(int)
y_true = test_generator.labels

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
f1 = f1_score(y_true, y_pred_labels)

# Append accuracy and loss history for overall plot
overall_accuracy_history.append(accuracy)
overall_loss_history.append(loss)
# Append validation accuracy and loss history for overall plot
overall_val_accuracy_history.append(history.history['val_accuracy'])
overall_val_loss_history.append(history.history['val_loss'])
# Append sensitivity, specificity, and F1 score to their respective lists
sensitivities.append(sensitivity)
specificities.append(specificity)
f1_scores.append(f1)


# Calculate the mean accuracy and loss scores
mean_accuracy = np.mean(accuracy_scores)
mean_loss = np.mean(loss_scores)
mean_sensitivity = np.mean(sensitivities)
mean_specificity = np.mean(specificities)
mean_f1 = np.mean(f1_scores)

print('Mean Test Accuracy:', mean_accuracy)
print('Mean Test Loss:', mean_loss)
print('Mean Sensitivity:', mean_sensitivity)
print('Mean Specificity:', mean_specificity)
print('Mean F1 Score:', mean_f1)


for fold, history in enumerate(history_list, 1):  # Iterate over history_list
    plot_accuracy_fold(history)
    plot_loss_fold(history)


plot_accuracy_overall(overall_accuracy_history,overall_val_accuracy_history)
plot_loss_overall(overall_loss_history,overall_val_loss_history)
