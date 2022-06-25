# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Flatten


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
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)



def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')



    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000
    Max_Epochs = 30

    #read and parse dataset
    with open ('sarcasm.json','r') as f:
        datastore= json.load(f)

    sentences = []
    labels = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    #split dataset

    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]

    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    # Fit your tokenizer with training data

    #declare tokenizer
    tokenizer =  Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    # word_index = tokenizer.word_index

    #tokenize and padded training
    training_sequences= tokenizer.texts_to_sequences(training_sentences)
    training_padded= pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    #tokenize and padded testing

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    #convert labels to numpy

    training_labels= np.array(training_labels)
    testing_labels = np.array(testing_labels)

    #construct model
    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    #fit model
    model.fit(training_padded, training_labels, epochs= Max_Epochs,
              validation_data=(testing_padded, testing_labels),
              verbose= 1,
              callbacks=[myCallback(monitor='accuracy', min_perform= 0.76), myEarlyStop])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
