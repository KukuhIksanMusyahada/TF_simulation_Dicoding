# ======================================================================================================
# PROBLEM A3 
#
# Build a classifier for the Human or Horse Dataset with Transfer Learning. 
# The test will expect it to classify binary classes.
# Note that all the layers in the pre-trained model are non-trainable.
# Do not use lambda layers in your model.
#
# The horse-or-human dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
# Inception_v3, pre-trained model used in this problem is developed by Google.
#
# Desired accuracy and validation_accuracy > 93%.
# =======================================================================================================
import os
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_perform= 0.84):
        self.min_perform= min_perform
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None) and (logs.get('val_accuracy') is not None):
            if (logs.get('accuracy')> self.min_perform) and (logs.get('val_accuracy')> self.min_perform):
                print(f"\nReached {self.min_perform}% accuracy so cancelling training!")
                self.model.stop_training = True

def solution_A3():
    if os.path.isfile('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5') == False:
        inceptionv3='https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        urllib.request.urlretrieve(inceptionv3, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(
        input_shape=(150,150,3),
        include_top=False,
        weights=None,
    )
    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False


    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

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



    x = layers.Flatten()(last_output)
    x = layers.Dense(524, activation= 'relu')(x)
    x = layers.Dense(316, activation= 'relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_dir = 'data/horse-or-human'
    validation_dir = 'data/validation-horse-or-human'

    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory= train_dir,
        batch_size=35,
        class_mode='binary',
        target_size=(150,150)
    )
    val_datagen = ImageDataGenerator(
        rescale=1 / 255,
    )

    val_generator = val_datagen.flow_from_directory(
        directory=validation_dir,
        batch_size=35,
        class_mode='binary',
        target_size=(150, 150)
    )

    model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=100,
                        verbose=2,
                        callbacks= myCallback(min_perform=0.93)
                        )

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A3()
    model.save("model_A3.h5")
