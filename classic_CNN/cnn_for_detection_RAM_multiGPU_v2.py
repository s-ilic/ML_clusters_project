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

from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import layers


mirrored_strategy = tf.distribute.MirroredStrategy()

# Path variables
path_hc = "/home/users/ilic/ML/SDSS_image_data_unsorted"
path_nc = "/home/users/ilic/ML/SDSS_image_data_unsorted/randoms"

# List all training/validation images
all_hc = ['%s/%s' % (path_hc, f) for f in os.listdir(path_hc) if '.jpeg' in f]
all_nc = ['%s/%s' % (path_nc, f) for f in os.listdir(path_nc) if '.jpeg' in f]
# len(all_hc) = 14046
# len(all_nc) = 14047
np.random.shuffle(all_hc)
np.random.shuffle(all_nc)

# How many training/validation you want
batch_size = 16
n_want_train = batch_size * 300 * 3
n_want_valid = batch_size * 300
n_want_test = batch_size * 300

# Select images randomly
final_train = (
    all_hc[:(n_want_train // 2)] +
    all_nc[:(n_want_train // 2)]
)
np.random.shuffle(final_train)
ix = n_want_train // 2
final_valid = (
    all_hc[ix:ix+(n_want_valid // 2)] +
    all_nc[ix:ix+(n_want_valid // 2)]
)
np.random.shuffle(final_valid)
ix += n_want_valid // 2
final_test = (
    all_hc[ix:ix+(n_want_test // 2)] +
    all_nc[ix:ix+(n_want_test // 2)]
)
np.random.shuffle(final_test)

# Requested image properties
# target_size = (2048, 2048)
target_size = (1024, 1024)
shift_sig = 100

with mirrored_strategy.scope():

    model=Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    model.compile(
        # loss='mean_absolute_error',
        loss='binary_crossentropy',
        # optimizer='adam',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy'],
    )
    model.summary()

'''
with mirrored_strategy.scope():

    inputs = keras.Input(shape=(target_size[0], target_size[1], 3))

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    previous_block_activation = x  # Set aside residual

    for ix, size in enumerate([128, 256, 512, 728]):

        if ix != 0:
            x = layers.Activation("relu")(x)
            x = layers.Dropout(0.2)(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    # model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.compile(
        # loss='mean_absolute_error',
        loss='binary_crossentropy',
        optimizer='adam',
        # optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    model.summary()
'''

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

# prefix = "detect_1024x1024_shift100_Xception"
prefix = "detect_1024x1024_shift100_DeepCNN"

np.savez(
    "saved_weights/%s_ixs" % prefix,
    final_train=final_train,
    final_valid=final_valid,
    final_test=final_test,
)

log_filename = "saved_weights/%s.log" % prefix
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)

chk_filename = "saved_weights/%s" % prefix
chk_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
)

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_valid, Y_valid),
    validation_batch_size=batch_size,
    callbacks=[log_cb, chk_cb],
)


'''
model.save_weights('saved_weights/weights_%s.h5' % suff)
eval_test = model.evaluate(X_test, Y_test)
np.savez(
    'saved_weights/hist_%s' % suff,
    hist=history.history,
    eval_test=eval_test,
    final_train=final_train,
    final_valid=final_valid,
    final_test=final_test,
)

while True:
    history2 = model.fit(
        X_train,
        Y_train,
        epochs=20,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_valid, Y_valid),
        validation_batch_size=batch_size,
        callbacks=[history],
    )
    model.save_weights('saved_weights/weights_%s.h5' % suff)
    eval_test = model.evaluate(X_test, Y_test)
    np.savez(
        'saved_weights/hist_%s' % suff,
        hist=history.history,
        eval_test=eval_test,
        final_train=final_train,
        final_valid=final_valid,
        final_test=final_test,
    )
'''


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

X_toto = img_to_array(
    load_img(
        '/home/users/ilic/ML/SDSS_image_data/valid/has_cluster/10078.jpeg',
        color_mode='rgb',
        target_size=target_size,
        interpolation='nearest',
    ),
    dtype='uint8',
)
activations = activation_model.predict(X_toto[None, :, :, :])
for ix, lay in enumerate(activations):
    sh = lay.shape
    if len(sh) == 4:
        if sh[-1] == 3:
            nl, nc = 1, 3
        elif sh[-1] == 32:
            nl, nc = 4, 8
        elif sh[-1] == 64:
            nl, nc = 8, 8
        elif sh[-1] == 128:
            nl, nc = 8, 16
        elif sh[-1] == 256:
            nl, nc = 16, 16
        elif sh[-1] == 512:
            nl, nc = 16, 32
        elif sh[-1] == 728:
            nl, nc = 14, 52
        elif sh[-1] == 1024:
            nl, nc = 32, 32
        mat_to_plot = np.zeros((sh[1] * nl, sh[2] * nc))
        for i in range(nl):
            for j in range(nc):
                mat_to_plot[i*sh[1]:(i+1)*sh[1], j*sh[2]:(j+1)*sh[2]] = (
                    lay[0, :, :, i * nc + j]
                )
        plt.matshow(mat_to_plot, cmap='viridis')
        plt.title(name_layers[ix])
        plt.savefig('layer_%s.png' % (ix + 1), dpi=600)
        plt.clf()
        plt.matshow(np.log10(np.abs(mat_to_plot)), cmap='viridis')
        plt.title(name_layers[ix])
        plt.savefig('layer_%s_log10.png' % (ix + 1), dpi=600)
        plt.clf()



'''
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
'''
