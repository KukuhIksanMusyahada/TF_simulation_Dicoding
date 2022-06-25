# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense


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


def solution_B2():
    #Load_data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels)= fashion_mnist.load_data()
    #Normalize_data
    training_images = training_images/255.0
    test_images = test_images/255.0
    max_epoch= 20
    #Build Model
    model= Sequential([Flatten(input_shape=(28,28)),
                       Dense(32, activation='relu'),
                       Dense(10, activation='softmax')])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images,training_labels, validation_data=(test_images, test_labels),
              epochs= max_epoch, verbose= 2,
              callbacks= myCallback(monitor='accuracy', min_perform=0.85))

    # YOUR CODE HERE

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B2()
    model.save("model_B2.h5")


