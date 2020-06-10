import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from multiprocessing import Pool

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D

mirrored_strategy = tf.distribute.MirroredStrategy()

# Path variables
pathData="/mnt/local-scratch/u/ilic"
path_train = pathData + "/data/train"
path_valid = pathData + "/data/valid"
path_test = pathData + "/data/test"

# List all training/validation images
all_train_hc = np.array(['%s/has_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'has_cluster'))])
all_train_nc = np.array(['%s/no_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'no_cluster'))])
all_valid_hc = np.array(['%s/has_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'has_cluster'))])
all_valid_nc = np.array(['%s/no_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'no_cluster'))])
all_test_hc = np.array(['%s/has_cluster/%s' % (path_test, p) for p in os.listdir('%s/%s' % (path_test, 'has_cluster'))])
all_test_nc = np.array(['%s/no_cluster/%s' % (path_test, p) for p in os.listdir('%s/%s' % (path_test, 'no_cluster'))])

# How many training/validation you want
n_want_train = 8426
n_want_valid = 2810
n_want_test = 2810
# n_want_train = 842
# n_want_valid = 280
# n_want_test = 280
print("You asked for %s training images (out of %s available)" % (n_want_train, len(all_train_hc) + len(all_train_nc)))
print("You asked for %s validation images (out of %s available)" % (n_want_valid, len(all_valid_hc) + len(all_valid_nc)))
print("You asked for %s test images (out of %s available)" % (n_want_test, len(all_test_hc) + len(all_test_nc)))

# Select images randomly
ix_train_hc = np.argsort(np.random.rand(len(all_train_hc)))[:(n_want_train // 2)]
ix_train_nc = np.argsort(np.random.rand(len(all_train_nc)))[:(n_want_train // 2)]
final_train = list(all_train_hc[ix_train_hc]) + list(all_train_nc[ix_train_nc])
ix_valid_hc = np.argsort(np.random.rand(len(all_valid_hc)))[:(n_want_valid // 2)]
ix_valid_nc = np.argsort(np.random.rand(len(all_valid_nc)))[:(n_want_valid // 2)]
final_valid = list(all_valid_hc[ix_valid_hc]) + list(all_valid_nc[ix_valid_nc])
ix_test_hc = np.argsort(np.random.rand(len(all_test_hc)))[:(n_want_test // 2)]
ix_test_nc = np.argsort(np.random.rand(len(all_test_nc)))[:(n_want_test // 2)]
final_test = list(all_test_hc[ix_test_hc]) + list(all_test_nc[ix_test_nc])


# Requested image properties
target_size = (1024, 1024)
shift_sig = 100

# Some variables
batch_size = 64
dropoutpar = 0.5
nb_dense = 64

with mirrored_strategy.scope():

    model=Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    # model.add(Dense(nb_dense))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropoutpar))
    # model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.summary()


# Read the images
X_train = np.zeros((n_want_train, target_size[0], target_size[1], 3), dtype='uint8')
Y_train = np.zeros((n_want_train))
for i in tqdm(range(n_want_train)):
    shift = tuple((np.random.randn(2) * shift_sig).astype('int'))
    X_train[i, :, :, :] = np.roll(img_to_array(
        load_img(
            final_train[i],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='uint8',
    ), shift, axis=(0, 1))
    if i < (n_want_train / 2):
        Y_train[i] = 1.
X_valid = np.zeros((n_want_valid, target_size[0], target_size[1], 3), dtype='uint8')
Y_valid = np.zeros((n_want_valid))
for i in tqdm(range(n_want_valid)):
    X_valid[i, :, :, :] = img_to_array(
        load_img(
            final_valid[i],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='uint8',
    )
    if i < (n_want_valid / 2):
        Y_valid[i] = 1.
X_test = np.zeros((n_want_test, target_size[0], target_size[1], 3), dtype='uint8')
Y_test = np.zeros((n_want_test))
for i in tqdm(range(n_want_test)):
    X_test[i, :, :, :] = img_to_array(
        load_img(
            final_test[i],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='uint8',
    )
    if i < (n_want_test / 2):
        Y_test[i] = 1.

# np.savez(
#     "all_sets",
#     X_train=X_train,
#     Y_train=Y_train,
#     X_valid=X_valid,
#     Y_valid=Y_valid,
#     X_test=X_test,
#     Y_test=Y_test,
# )

# steps_per_epoch = int(n_train  // batch_size)
# validation_steps = int(n_valid // batch_size)

# sys.exit()

# model.load_weights('detect_weights_ns128.h5')
# model.load_weights('detect_weights_shift_ns128.h5')

sys.exit()

# history = model.fit_generator(
history = model.fit(
    X_train,
    Y_train,
    epochs=2,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_valid, Y_valid),
    validation_batch_size=batch_size,
)

while True:
    history2 = model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_valid, Y_valid),
        validation_batch_size=batch_size,
        callbacks=[history],
    )
    model.save_weights('detect_weights_shift_ns128_v2.h5')
    eval_test = model.evaluate(X_test, Y_test)
    np.savez('hist_shift_ns128_v2', hist=history.history, eval_test=eval_test)


sys.exit()

#################################################################################
#################################################################################
#################################################################################

from tensorflow.keras import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

layer_outputs = [layer.output for layer in model.layers]
name_layers = [layer.name for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X_train[:1, :, :, :])

im = array_to_img(X_train[0, :, :, :])


for ix, lay in enumerate(activations[:8]):
    sh = lay.shape
    mat_to_plot = np.zeros((sh[1] * 4, sh[2] * 8))
    for i in range(4):
        print(i)
        for j in range(8):
            mat_to_plot[i*sh[1]:(i+1)*sh[1], j*sh[2]:(j+1)*sh[2]] = (
                lay[0, :, :, i * 8 + j]
            )
    plt.matshow(mat_to_plot, cmap='viridis')
    plt.title(name_layers[ix])
    plt.savefig('layer_%s.png' % (ix + 1), dpi=300)
    plt.clf()

preds = model.predict(X_train)
for i in tqdm(range(1, 1000)):
    if preds[i, 0] == 1.:
        activations_tmp = activation_model.predict(X_train[i:(i+1), :, :, :])
        for ix in range(11):
            activations[ix] += activations_tmp[ix]

