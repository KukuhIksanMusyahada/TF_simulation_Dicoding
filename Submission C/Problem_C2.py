# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten

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


def solution_C2():
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels),(test_images, test_labels) = mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    training_images = training_images/training_images.max()
    test_images = test_images / test_images.max()
    max_epochs= 20
    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model= Sequential([Flatten(input_shape=(28,28)),
                       Dense(32, activation='relu'),
                       Dense(10, activation='softmax')])

    # COMPILE MODEL HERE
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics= 'accuracy')
    # TRAIN YOUR MODEL HERE
    model.fit(training_images, training_labels,
              validation_data=(test_images,test_labels),
              epochs= max_epochs,
              callbacks= [myCallback(monitor='accuracy', min_perform= 0.92), myEarlyStop])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
