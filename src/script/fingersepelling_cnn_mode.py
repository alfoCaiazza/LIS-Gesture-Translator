import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Precision, Recall, SparseCategoricalAccuracy

data_directory = 'src/data/processed'

# To avoid OOM errors, setting GPU Memory Consuption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(f"GPU: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True) # Keeping the use of memory limited to prevent errors

# Automatically creates a dataset form the referred directory. Load the full dataset, shuffle = True ensures randomness
dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    image_size=(256, 256),
    batch_size=32,
    shuffle=True,
    seed=123
)

# In order to iterare the element we must use the iterative method
data_iterator = dataset.as_numpy_iterator()

def FingerSpellingModel(input_shape=(256, 256, 3), num_classes=22):
    model = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
