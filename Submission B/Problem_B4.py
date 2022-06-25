# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


# this following function was snipped from Coursera Assignment
# at TensorFlow Developer Specialization
# 1. remove_stopwords function
# 2. fit_tokenizer
# 3. seq_and_pad
# 4. Tokenize_labels

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



def fit_tokenizer(train_sentences, num_words, oov_token):

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    tokenizer.fit_on_texts(train_sentences)

    return tokenizer


def seq_and_pad(sentences, tokenizer, padding, maxlen, trunc_type):

    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=trunc_type)

    return padded_sequences


def tokenize_labels(all_labels, split_labels):

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(all_labels)
    label_seq = label_tokenizer.texts_to_sequences(split_labels)
    label_seq_np = (np.array(label_seq) - 1)

    return label_seq_np


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    # bbc = parse_data(bbc, target_column='text')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8
    max_epoch = 30

    # YOUR CODE HERE
    train, test, train_label, test_label = train_test_split(bbc['text'], bbc['category'], train_size = training_portion, shuffle= False)
    tokenizer = fit_tokenizer(train, vocab_size, oov_token=oov_tok)
    word_index = tokenizer.word_index

    train_padded_seq = seq_and_pad(train, tokenizer, padding_type, max_length, trunc_type)
    val_padded_seq = seq_and_pad(test, tokenizer, padding_type, max_length, trunc_type)

    train_label_seq = tokenize_labels(bbc['category'], train_label)
    val_label_seq = tokenize_labels(bbc['category'], test_label)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics= ['accuracy'])
    model.fit(train_padded_seq, train_label_seq,
              epochs= max_epoch,
              validation_data= (val_padded_seq, val_label_seq),
              verbose= 1,
              callbacks= [myCallback(monitor='accuracy', min_perform=0.93), myEarlyStop])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
