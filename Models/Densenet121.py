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
data = pd.read_excel('HRDC Hypertensive Retinopathy Classification Training Labels.xlsx', engine='openpyxl')
image_files = data['Image'].tolist()
labels = data['Hypertensive Retinopathy'].tolist()

data['Hypertensive Retinopathy'] = data['Hypertensive Retinopathy'].astype(str)

# Create data generator for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory='1-Training Set',
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
    directory='1-Training Set',
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
    directory='Testing Set',
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



   
   
    "Epoch 1/20\n",
      "18/18 [==============================] - 406s 19s/step - loss: 0.7790 - accuracy: 0.6175 - val_loss: 0.7366 - val_accuracy: 0.2817\n",
      "Epoch 2/20\n",
      "18/18 [==============================] - 304s 17s/step - loss: 0.6887 - accuracy: 0.6544 - val_loss: 0.7925 - val_accuracy: 0.2606\n",
      "Epoch 3/20\n",
      "18/18 [==============================] - 302s 17s/step - loss: 0.6558 - accuracy: 0.6544 - val_loss: 0.7034 - val_accuracy: 0.4930\n",
      "Epoch 4/20\n",
      "18/18 [==============================] - 293s 16s/step - loss: 0.6413 - accuracy: 0.6632 - val_loss: 0.7176 - val_accuracy: 0.3592\n",
      "Epoch 5/20\n",
      "18/18 [==============================] - 2085s 122s/step - loss: 0.6314 - accuracy: 0.6754 - val_loss: 0.7343 - val_accuracy: 0.2606\n",
      "Epoch 6/20\n",
      "18/18 [==============================] - 287s 16s/step - loss: 0.6251 - accuracy: 0.6667 - val_loss: 0.8206 - val_accuracy: 0.2606\n",
      "Epoch 7/20\n",
      "18/18 [==============================] - 283s 16s/step - loss: 0.6228 - accuracy: 0.6596 - val_loss: 1.0241 - val_accuracy: 0.2606\n",
      "Epoch 8/20\n",
      "18/18 [==============================] - 294s 16s/step - loss: 0.6401 - accuracy: 0.6737 - val_loss: 0.8930 - val_accuracy: 0.2606\n",
      "Epoch 9/20\n",
      "18/18 [==============================] - 396s 22s/step - loss: 0.6087 - accuracy: 0.6737 - val_loss: 0.8899 - val_accuracy: 0.2606\n",
      "Epoch 10/20\n",
      "18/18 [==============================] - 463s 26s/step - loss: 0.6090 - accuracy: 0.6632 - val_loss: 0.8264 - val_accuracy: 0.2606\n",
      "Epoch 11/20\n",
      "18/18 [==============================] - 446s 25s/step - loss: 0.6041 - accuracy: 0.6860 - val_loss: 0.9570 - val_accuracy: 0.2606\n",
      "Epoch 12/20\n",
      "18/18 [==============================] - 64217s 3776s/step - loss: 0.5998 - accuracy: 0.6947 - val_loss: 0.7885 - val_accuracy: 0.2606\n",
      "Epoch 13/20\n",
      "18/18 [==============================] - 322s 18s/step - loss: 0.5921 - accuracy: 0.6807 - val_loss: 0.7986 - val_accuracy: 0.2817\n",
      "Epoch 14/20\n",
      "18/18 [==============================] - 285s 16s/step - loss: 0.6054 - accuracy: 0.6754 - val_loss: 1.1269 - val_accuracy: 0.2606\n",
      "Epoch 15/20\n",
      "18/18 [==============================] - 295s 16s/step - loss: 0.5878 - accuracy: 0.6807 - val_loss: 0.9830 - val_accuracy: 0.2606\n",
      "Epoch 16/20\n",
      "18/18 [==============================] - 299s 17s/step - loss: 0.6256 - accuracy: 0.6807 - val_loss: 0.8050 - val_accuracy: 0.2676\n",
      "Epoch 17/20\n",
      "18/18 [==============================] - 279s 16s/step - loss: 0.5946 - accuracy: 0.7035 - val_loss: 0.7738 - val_accuracy: 0.2606\n",
      "Epoch 18/20\n",
      "18/18 [==============================] - 297s 16s/step - loss: 0.6164 - accuracy: 0.6649 - val_loss: 0.8486 - val_accuracy: 0.2606\n",
      "Epoch 19/20\n",
      "18/18 [==============================] - 290s 16s/step - loss: 0.6356 - accuracy: 0.6684 - val_loss: 1.1022 - val_accuracy: 0.2606\n",
      "Epoch 20/20\n",
      "18/18 [==============================] - 297s 16s/step - loss: 0.6062 - accuracy: 0.6737 - val_loss: 0.9836 - val_accuracy: 0.2606\n",
      "Found 712 validated image filenames belonging to 2 classes.\n",
      "23/23 [==============================] - 55s 2s/step - loss: 0.7026 - accuracy: 0.5899\n",
      "Test Loss: 0.7026246190071106\n",
      "Test Accuracy: 0.5898876190185547\n"
     ]
    }
   ],
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
    "    shuffle=False,\n",
    "    class_mode='binary',\n",
    "    target_size=(224, 224)\n",
    ")\n",
    "\n",
    "loss, accuracy = model.evaluate(test_generator)\n",
    "print('Test Loss:', loss)\n",
    "print('Test Accuracy:', accuracy)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   
