import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from astropy.io import fits
# from multiprocessing import Pool

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

# Get cluster info
hdul = fits.open(pathData+'/redmapper_dr8_public_v6.3_catalog.fits')
clu_full_data = hdul[1].data
ixs_trhc = [int(f[:-5]) for f in os.listdir('%s/%s' % (path_train, 'has_cluster'))]
z_trhc = [clu_full_data['z_lambda'][clu_full_data['ID']==ix][0] for ix in ixs_trhc]
ixs_vhc = [int(f[:-5]) for f in os.listdir('%s/%s' % (path_valid, 'has_cluster'))]
z_vhc = [clu_full_data['z_lambda'][clu_full_data['ID']==ix][0] for ix in ixs_vhc]
ixs_thc = [int(f[:-5]) for f in os.listdir('%s/%s' % (path_test, 'has_cluster'))]
z_thc = [clu_full_data['z_lambda'][clu_full_data['ID']==ix][0] for ix in ixs_thc]

# List all training/validation images
all_train_hc = np.array(['%s/has_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'has_cluster'))])
all_train_nc = np.array(['%s/no_cluster/%s' % (path_train, p) for p in os.listdir('%s/%s' % (path_train, 'no_cluster'))])
all_valid_hc = np.array(['%s/has_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'has_cluster'))])
all_valid_nc = np.array(['%s/no_cluster/%s' % (path_valid, p) for p in os.listdir('%s/%s' % (path_valid, 'no_cluster'))])
all_test_hc = np.array(['%s/has_cluster/%s' % (path_test, p) for p in os.listdir('%s/%s' % (path_test, 'has_cluster'))])
all_test_nc = np.array(['%s/no_cluster/%s' % (path_test, p) for p in os.listdir('%s/%s' % (path_test, 'no_cluster'))])

# How many training/validation you want
n_want_train_hc = 8320
n_want_train_nc = 1024
n_want_valid_hc = 2048
n_want_valid_nc = 256
n_want_test_hc = 2048
n_want_test_nc = 256
print("You asked for %s training images with clusters (out of %s available)" % (n_want_train_hc, len(all_train_hc)))
print("You asked for %s training images without clusters (out of %s available)" % (n_want_train_nc, len(all_train_nc)))
print("You asked for %s validation images with clusters (out of %s available)" % (n_want_valid_hc, len(all_valid_hc)))
print("You asked for %s validation images without clusters (out of %s available)" % (n_want_valid_nc, len(all_valid_nc)))
print("You asked for %s test images with clusters (out of %s available)" % (n_want_test_hc, len(all_test_hc)))
print("You asked for %s test images without clusters (out of %s available)" % (n_want_test_nc, len(all_test_nc)))

# Select images randomly
ix_train_hc = np.argsort(np.random.rand(len(all_train_hc)))
ix_train_nc = np.argsort(np.random.rand(len(all_train_nc)))
ix_valid_hc = np.argsort(np.random.rand(len(all_valid_hc)))
ix_valid_nc = np.argsort(np.random.rand(len(all_valid_nc)))
ix_test_hc = np.argsort(np.random.rand(len(all_test_hc)))
ix_test_nc = np.argsort(np.random.rand(len(all_test_nc)))


# Requested image properties
target_size = (1024, 1024)

# Read the images
## Train
X_train = np.zeros(
    (
        n_want_train_hc + n_want_train_nc,
        target_size[0],
        target_size[1],
        3
    ),
    dtype='int8'
)
Y_train = np.zeros((n_want_train_hc + n_want_train_nc, 2))
for i in tqdm(range(n_want_train_hc)):
    X_train[i, :, :, :] = img_to_array(
        load_img(
            all_train_hc[ix_train_hc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
    Y_train[i, 0] = 1.
    Y_train[i, 1] = z_trhc[ix_train_hc[i]]
for i in tqdm(range(n_want_train_nc)):
    X_train[n_want_train_hc + i, :, :, :] = img_to_array(
        load_img(
            all_train_nc[ix_train_nc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
## Valid
X_valid = np.zeros(
    (
        n_want_valid_hc + n_want_valid_nc,
        target_size[0],
        target_size[1],
        3
    ),
    dtype='int8'
)
Y_valid = np.zeros((n_want_valid_hc + n_want_valid_nc, 2))
for i in tqdm(range(n_want_valid_hc)):
    X_valid[i, :, :, :] = img_to_array(
        load_img(
            all_valid_hc[ix_valid_hc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
    Y_valid[i, 0] = 1.
    Y_valid[i, 1] = z_vhc[ix_valid_hc[i]]
for i in tqdm(range(n_want_valid_nc)):
    X_valid[n_want_valid_hc + i, :, :, :] = img_to_array(
        load_img(
            all_valid_nc[ix_valid_nc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
## Test
X_test = np.zeros(
    (
        n_want_test_hc + n_want_test_nc,
        target_size[0],
        target_size[1],
        3
    ),
    dtype='int8'
)
Y_test = np.zeros((n_want_test_hc + n_want_test_nc, 2))
for i in tqdm(range(n_want_test_hc)):
    X_test[i, :, :, :] = img_to_array(
        load_img(
            all_test_hc[ix_test_hc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )
    Y_test[i, 0] = 1.
    Y_test[i, 1] = z_thc[ix_test_hc[i]]
for i in tqdm(range(n_want_test_nc)):
    X_test[n_want_test_hc + i, :, :, :] = img_to_array(
        load_img(
            all_test_nc[ix_test_nc[i]],
            color_mode='rgb',
            target_size=target_size,
            interpolation='nearest',
        ),
        dtype='int8',
    )

# Some variables
batch_size = 32
dropoutpar = 0.5
nb_dense = 64

with mirrored_strategy.scope():

    model=Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(nb_dense))
    model.add(Activation('relu'))
    model.add(Dropout(dropoutpar))
    model.add(Dense(2, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.summary()

# steps_per_epoch = int(n_train  // batch_size)
# validation_steps = int(n_valid // batch_size)

# sys.exit()

# model.load_weights('detect_weights.h5') 

# history = model.fit_generator(
history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_valid, Y_valid),
    validation_batch_size=batch_size,
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
