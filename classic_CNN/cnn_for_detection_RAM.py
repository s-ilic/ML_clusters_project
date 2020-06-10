import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import  BatchNormalization
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D

# Path variables
pathData="/mnt/local-scratch/u/ilic"
path_train = pathData + "/data/train"
path_valid = pathData + "/data/valid"

# List all training/validation images
all_train_hc = np.array(['%s/has_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'has_cluster'))])
all_train_nc = np.array(['%s/no_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'no_cluster'))])
all_valid_hc = np.array(['%s/has_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'has_cluster'))])
all_valid_nc = np.array(['%s/no_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'no_cluster'))])

# How many training/validation you want
n_want_train = 11236
n_want_valid = 2810
# n_want_train = 1120
# n_want_valid = 280
print("You asked for %s training images (out of %s available)" % (n_want_train, len(all_train_hc) + len(all_train_nc)))
print("You asked for %s validation images (out of %s available)" % (n_want_valid, len(all_valid_hc) + len(all_valid_nc)))

# Select images randomly
ix_train_hc = np.argsort(np.random.rand(len(all_train_hc)))[:(n_want_train // 2)]
ix_train_nc = np.argsort(np.random.rand(len(all_train_nc)))[:(n_want_train // 2)]
ix_valid_hc = np.argsort(np.random.rand(len(all_valid_hc)))[:(n_want_valid // 2)]
ix_valid_nc = np.argsort(np.random.rand(len(all_valid_nc)))[:(n_want_valid // 2)]
final_train = list(all_train_hc[ix_train_hc]) + list(all_train_nc[ix_train_nc])
final_valid = list(all_valid_hc[ix_valid_hc]) + list(all_valid_nc[ix_valid_nc])

# Requested image properties
target_size = (1024, 1024)

# Read the images
X_train = np.zeros((n_want_train, target_size[0], target_size[1], 3), dtype='int8')
Y_train = np.zeros((n_want_train))
for i in tqdm(range(n_want_train)):
    X_train[i, :, :, :] = img_to_array(
        load_img(
            final_train[i],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
    if i < (n_want_train / 2):
        Y_train[i] = 1.
X_valid = np.zeros((n_want_valid, target_size[0], target_size[1], 3), dtype='int8')
Y_valid = np.zeros((n_want_valid))
for i in tqdm(range(n_want_valid)):
    X_valid[i, :, :, :] = img_to_array(
        load_img(
            final_valid[i],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
    if i < (n_want_valid / 2):
        Y_valid[i] = 1.

# Some variables
batch_size = 16
dropoutpar = 0.5
nb_dense = 64

model=Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(target_size[0], target_size[1], 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(nb_dense, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(dropoutpar))
# model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# steps_per_epoch = int(n_train  // batch_size)
# validation_steps = int(n_valid // batch_size)

# sys.exit()

# history = model.fit_generator(
history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_valid, Y_valid),
    # validation_freq=[5, 10],
)



'''
sys.exit()

######################

import tqdm

all_vhc = os.listdir(path_valid + '/has_cluster')
pred_vhc = []
for f in tqdm.tqdm(all_vhc):
    img = load_img(path_valid + '/has_cluster/%s' % f)
    x = img_to_array(img.resize((1024,1024)))
    x = x.reshape((1,) + x.shape)
    pred_vhc.append(model.predict(x)[0][0])

all_vnc = os.listdir(path_valid + '/no_cluster')
pred_vnc = []
for f in tqdm.tqdm(all_vnc):
    img = load_img(path_valid + '/no_cluster/%s' % f)
    x = img_to_array(img.resize((1024,1024)))
    x = x.reshape((1,) + x.shape)
    pred_vnc.append(model.predict(x)[0][0])

all_thc = os.listdir(path_train + '/has_cluster')
pred_thc = []
for f in tqdm.tqdm(all_thc):
    img = load_img(path_train + '/has_cluster/%s' % f)
    x = img_to_array(img.resize((1024,1024)))
    x = x.reshape((1,) + x.shape)
    pred_thc.append(model.predict(x)[0][0])

all_tnc = os.listdir(path_train + '/no_cluster')
pred_tnc = []
for f in tqdm.tqdm(all_tnc):
    img = load_img(path_train + '/no_cluster/%s' % f)
    x = img_to_array(img.resize((1024,1024)))
    x = x.reshape((1,) + x.shape)
    pred_tnc.append(model.predict(x)[0][0])

np.savez(
    "predic_1",
    all_vhc=all_vhc,
    pred_vhc=pred_vhc,
    all_vnc=all_vnc,
    pred_vnc=pred_vnc,
    all_thc=all_thc,
    pred_thc=pred_thc,
    all_tnc=all_tnc,
    pred_tnc=pred_tnc,
)
'''
