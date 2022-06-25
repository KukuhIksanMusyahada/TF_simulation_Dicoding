# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network model using the Daily Min Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 0.19 on the normalized dataset.
# ============================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib

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
                if (logs.get('mae')< self.min_perform):
                    print(f"\nReached {self.min_perform} min_perform so cancelling training!")
                    self.model.stop_training = True

myEarlyStop= tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_C5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    urllib.request.urlretrieve(data_url, 'daily-min-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-min-temperatures.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(float(row[1]))
            time_step.append(step)
            step=step + 1

    series=  np.array(temps)

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series)
    max=np.max(series)
    series -= min
    series /= max
    time=np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time=2500

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time]
    x_valid = series[split_time]

    # DO NOT CHANGE THIS CODE
    window_size=64
    batch_size=256
    shuffle_buffer_size=1000
    max_epochs = 100

    train_set=windowed_dataset(
        x_train, window_size, batch_size, shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=[None, 1]),
        # tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['mae'])
    model.fit(train_set, epochs=max_epochs,
              callbacks=[myCallback(monitor='mae', min_perform=0.18), myEarlyStop],
              )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_C5()
    model.save("model_C5.h5")
