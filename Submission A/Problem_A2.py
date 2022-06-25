# =====================================================================================
# PROBLEM A2 
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes. 
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
import numpy as np

from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop



class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_perform= 0.84):
        self.min_perform= min_perform
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None) and (logs.get('val_accuracy') is not None):
            if (logs.get('accuracy')> self.min_perform) and (logs.get('val_accuracy')> self.min_perform):
                print(f"\nReached {self.min_perform}% accuracy so cancelling training!")
                self.model.stop_training = True


def solution_A2():
    train_file_path = os.path.join('.', 'horse-or-human.zip')
    val_file_path = os.path.join('.', 'validation-horse-or-human.zip')
    if os.path.isfile(train_file_path)==False:
        data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
        urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')

    if os.path.isfile(val_file_path)==False:
        data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
        urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    if len(os.listdir('./data'))==0:
        local_file = ['horse-or-human.zip','validation-horse-or-human.zip']
        zip_ref = zipfile.ZipFile(local_file[0], 'r')
        zip_ref.extractall('data/horse-or-human')

        zip_ref = zipfile.ZipFile(local_file[1], 'r')
        zip_ref.extractall('data/validation-horse-or-human')
        zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'
    train_datagen = ImageDataGenerator(
        rescale=1./255)
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=20,
                                                                  class_mode='binary',
                                                                  target_size=(150, 150))

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        # keras.layers.Dropout(.6),

        keras.layers.Flatten(),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        # keras.layers.Dropout(.5),
        keras.layers.Dense(32, activation='relu'),
        # keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=RMSprop(learning_rate= 0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              epochs=20,
              verbose=1,
              validation_data=validation_generator,
              callbacks= myCallback(min_perform= 0.85))

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A2()
    model.save("model_A2.h5")
