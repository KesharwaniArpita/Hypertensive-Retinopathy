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



   
   
   "source": [
    "\n",
    "import pandas as pd\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import DenseNet121\n",
    "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n",
    "from tensorflow.keras.models import Model\n",
    "\n",
    "# Load the DenseNet121 model without the top (classification) layers\n",
    "base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))\n",
    "\n",
    "# binary classification (hypertensive vs. normal)\n",
    "x = base_model.output\n",
    "x = GlobalAveragePooling2D()(x)\n",
    "x = Dense(1024, activation='relu')(x)\n",
    "predictions = Dense(1, activation='sigmoid')(x)  \n",
    "model = Model(inputs=base_model.input, outputs=predictions)\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Load the image filenames and labels from the Excel file\n",
    "data = pd.read_excel('C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')\n",
    "image_files = data['Image'].tolist()\n",
    "labels = data['Hypertensive Retinopathy'].tolist()\n",
    "\n",
    "data['Hypertensive Retinopathy'] = data['Hypertensive Retinopathy'].astype(str)\n",
    "\n",
    "# Create data generator for training and validation\n",
    "datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)\n",
    "\n",
    "train_generator = datagen.flow_from_dataframe(\n",
    "    dataframe=data,\n",
    "    directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',\n",
    "    x_col='Image',\n",
    "    y_col='Hypertensive Retinopathy',\n",
    "    subset='training',\n",
    "    batch_size=32,\n",
    "    shuffle=True,\n",
    "    class_mode='binary',\n",
    "    target_size=(224, 224)\n",
    ")\n",
    "\n",
    "valid_generator = datagen.flow_from_dataframe(\n",
    "    dataframe=data,\n",
    "    directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',\n",
    "    x_col='Image',\n",
    "    y_col='Hypertensive Retinopathy',\n",
    "    subset='validation',\n",
    "    batch_size=32,\n",
    "    shuffle=True,\n",
    "    class_mode='binary',\n",
    "    target_size=(224, 224)\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "model.fit(train_generator, validation_data=valid_generator, epochs=100)\n",
    "\n",
    "# Evaluate the model on the test set\n",
    "test_generator = datagen.flow_from_dataframe(\n",
    "    dataframe=data,\n",
    "    directory='C:/Users/DELL/Documents/IAS internship docs/IAS project/Hypertensive retinopathy/Dataset/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',\n",
    "    x_col='Image',\n",
    "    y_col='Hypertensive Retinopathy',\n",
    "    batch_size=32,\n",
    "
 
   
