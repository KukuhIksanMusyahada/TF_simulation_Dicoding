# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization

class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='accuracy',min_perform= 0.84):
        self.min_perform= min_perform
        self.monitor= monitor
    def on_epoch_end(self, epoch, logs={}):
        if self.monitor=='accuracy':
            if (logs.get('accuracy') is not None) and (logs.get('val_accuracy') is not None):
                if (logs.get('accuracy')> self.min_perform) and (logs.get('val_accuracy')> self.min_perform):
                    print(f"\nReached {self.min_perform}% min_perform so cancelling training!")
                    self.model.stop_training = True
        elif self.monitor=='loss':
            if (logs.get('loss') is not None):
                if (logs.get('loss')< self.min_perform):
                    print(f"\nReached {self.min_perform}% min_perform so cancelling training!")
                    self.model.stop_training = True
        elif self.monitor=='mae':
            if (logs.get('mae') is not None):
                if (logs.get('loss')< self.min_perform):
                    print(f"\nReached {self.min_perform}% min_perform so cancelling training!")
                    self.model.stop_training = True

myEarlyStop= tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        class_mode='categorical',
        batch_size=32,
        subset='training'
    )
    val_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=32,
        subset='validation'
    )
    Max_epochs=20



    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150,3)),
        BatchNormalization(),
        # MaxPooling2D(2, 2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator,epochs= Max_epochs,
              callbacks= [myCallback(monitor='accuracy', min_perform=0.84), myEarlyStop],
              verbose=1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B3()
    model.save("model_B3.h5")

