import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, MaxPooling2D, Add, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def mobile_hr_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Step 1: Input normalization of raw data
    x = inputs

    # Step 2: Function definition
    def conv_bn(x, filters, kernel_size=3, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Step 3: Kernel sizes and conv-batch norm
    x = conv_bn(x, filters=32)
    y = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(x)

    # Step 4: Depthwise Conv2D was used rather than Conv2D
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Step 5: Establishing the network with skip connections
    for _ in range(14):
        # Skip connection
        x = Add()([x, y])
        y = x

        # Depthwise Convolution
        x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Pointwise Convolution
        x = conv_bn(x, filters=32)

    # Step 6: Flattened layer and feature map extraction
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)

    # SVM Classifier
    svm_output = Dense(1, activation='linear')(x)

    # Create the Mobile-HR model
    model = Model(inputs=inputs, outputs=svm_output)

    return model

