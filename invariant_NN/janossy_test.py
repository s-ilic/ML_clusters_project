import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool
from itertools import combinations, permutations
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

mirrored_strategy = tf.distribute.MirroredStrategy()

NUM_TRAINING_EXAMPLES = 100000
NUM_TEST_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLES = 10000
NUM_EPOCHS_JANOSSY = 1000
NUM_EPOCHS_RNN = 1000
BASE_EMBEDDING_DIMENSION = 100
INFERENCE_PERMUTATIONS = 20


def construct_task_specific_output(task, input_sequence):
    if task == 'range':
        return np.max(input_sequence) - np.min(input_sequence)
    if task == 'sum':
        return np.sum(input_sequence)
    if task == 'mean':
        return np.mean(input_sequence)
    if task == 'unique_sum':
        return np.sum(np.unique(input_sequence))
    if task == 'unique_count':
        return np.size(np.unique(input_sequence))
    if task == 'variance':
        return np.var(input_sequence)
    if task == 'stddev':
        return np.std(input_sequence)


def janossy_text_input_construction(X, janossy_k):
    X_janossy = []
    for index in range(len(X)):
        temp = list(X[index])
        temp = [int(x) for x in temp]
        temp.sort()
        # temp = list(combinations(temp, janossy_k))
        temp = list(permutations(temp, janossy_k))
        temp = [list(x) for x in temp]
        X_janossy.append(temp)
    return np.array(X_janossy)


def text_dataset_construction(train_or_test, janossy_k, task):
    """ Data Generation """
    if train_or_test:
        num_examples = NUM_TRAINING_EXAMPLES
    else:
        num_examples = NUM_TEST_EXAMPLES

    train_length = sequence_length
    X = np.zeros((num_examples, train_length))
    output_X = np.zeros((num_examples, 1))
    for i in tqdm(range(num_examples), desc='Generating Training / Validation / Test Examples: '):
        for j in range(train_length):
            X[i, j] = np.random.randint(0, vocab_size)
            output_X[i, 0] = construct_task_specific_output(task, X[i])
    if janossy_k == 1:
        return X, output_X
    else:
        # Create Janossy Input
        X_janossy = janossy_text_input_construction(X, janossy_k)
        return X_janossy, output_X

vocab_size = 100
sequence_length = 5
janossy_k = 5
# task = 'sum'
task = 'mean'
num_epochs = NUM_EPOCHS_JANOSSY
num_neurons_in_f = 30
num_layers_in_rho = 1
num_neurons_in_rho = 100

X_train, Y_train = text_dataset_construction(1, janossy_k, task)
X_valid, Y_valid = text_dataset_construction(0, janossy_k, task)
X_test, Y_test = text_dataset_construction(0, janossy_k, task)

# input_dim = int(BASE_EMBEDDING_DIMENSION / janossy_k)
# input_dim_mod = input_dim * janossy_k

with mirrored_strategy.scope():

    inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    resh_inputs = tf.reshape(inputs, (-1, X_train.shape[2]))
    lay1 = Dense(
        num_neurons_in_f,
        input_shape=(X_train.shape[2],),
        activation="relu",
    )
    out_lay1 = lay1(resh_inputs)
    resh_out_lay1 = tf.reshape(out_lay1, (-1, X_train.shape[1], num_neurons_in_f))
    output = tf.math.reduce_mean(
        resh_out_lay1,
        axis=1,
    )
    rho_lays = []
    for i in range(num_layers_in_rho):
        rho_lays.append(Dense(num_neurons_in_rho, activation="relu"))
        output = rho_lays[-1](output)
    output = Dense(1)(output)

    model = keras.Model(inputs, output)
    model.compile(
        loss='mean_absolute_percentage_error',
        optimizer=Adam(
            learning_rate=0.0001,
        ),
        metrics=['accuracy'],
    )

model.summary()

nb_epoch=1000

history = model.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=2**16,
    #shuffle=True,
    validation_data=(X_valid, Y_valid),
    #validation_batch_size=batch_size,
)

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Valid loss')
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.yscale('log')
plt.legend()
plt.savefig('loss_janop.pdf')

# train_text(
#     vocabulary_size, => 100
#     BASE_EMBEDDING_DIMENSION, => 100
#     task, => 'sum'
#     model, => 'janossy'
#     num_layers, => 1
#     num_neurons, => 100
#     janossy_k, => 1
#     learning_rate, => 1e-4
#     batch_size, => 128
#     iteration => 1
# )

