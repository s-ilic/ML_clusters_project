import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from astropy.io import fits
from multiprocessing import Pool

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow.keras.backend as K

mirrored_strategy = tf.distribute.MirroredStrategy()

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
print("Variables in cluster catalog:")
print(list(clu_full_data.dtype.names))
print("Variables in member catalog:")
print(list(mem_full_data.dtype.names))

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
g = clu_num == np.bincount(clu_num).argmax()
clu_ids = clu_ids[g]
n_clus = len(clu_ids)
n_mem_max = clu_num[g].max()
np.random.shuffle(clu_ids)

# Build initial feature & label vectors
n_feat_per_gal = 3
n_labels_clus = 3

allX = np.zeros((n_clus, n_feat_per_gal * n_mem_max))# + 1))
allY = np.zeros((n_clus, n_labels_clus))
for i in tqdm(range(n_clus)):
    g = mem_full_data['ID'] == clu_ids[i]
    # n_mem = g.sum()
    # allX[i, 0] = n_mem
    #####
    ras = mem_full_data['RA'][g] / 180. * np.pi
    decs = mem_full_data['DEC'][g] / 180. * np.pi
    x2s = (np.cos(decs) * np.cos(ras))**2.
    y2s = (np.cos(decs) * np.sin(ras))**2.
    z2s = (np.sin(decs))**2.
    tmp = np.vstack((x2s, y2s, z2s)).T.flatten()
    # allX[i, 1:(n_mem * n_feat_per_gal)+1] = tmp
    allX[i, :] = tmp
    #####
    g = clu_full_data['ID'] == clu_ids[i]
    rac = clu_full_data['RA'][g] / 180. * np.pi
    decc = clu_full_data['DEC'][g] / 180. * np.pi
    x2c = (np.cos(decc) * np.cos(rac))**2.
    y2c = (np.cos(decc) * np.sin(rac))**2.
    z2c = (np.sin(decc))**2.
    allY[i, :] = np.array([x2c[0], y2c[0], z2c[0]])

# Split training and validation datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[:max_ix,:]
X_valid = allX[max_ix:,:]
Y_train = allY[:max_ix, :]
Y_valid = allY[max_ix:, :]

'''
tmp = np.load("/home/users/ilic/ML/data.npz")
X_train = tmp['X_train']
X_valid = tmp['X_valid']
Y_train = tmp['Y_train']
Y_valid = tmp['Y_valid']
'''

# inputs = keras.Input(shape=(n_feat_per_gal * n_mem_max,))
# lay1 = Dense(10, input_dim=n_feat_per_gal * n_mem_max // 3)
# xs = [inputs[:, i::3] for i in range(3)]
# xs = [lay1(x) for x in xs]
# xs = [layers.Activation("relu")(x) for x in xs]
# xs = layers.add(xs)
# xs = Dense(n_labels_clus)(xs)
# xs = layers.Activation("relu")(xs)
# model = keras.Model(inputs, xs)
# model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
# model.summary()

class myLinear(layers.Layer):

    def __init__(self, input_nfeats=1):
        super(myLinear, self).__init__()
        self.input_nfeats = input_nfeats

    def build(self, input_shape):
        alpha_init = tf.random_normal_initializer()
        beta_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(
            initial_value=alpha_init(
                shape=(self.input_nfeats,),
                dtype='float32',
            ),
            trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=beta_init(
                shape=(self.input_nfeats,),
                dtype='float32',
            ),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = inputs.get_shape()[0]
        num_point = inputs.get_shape()[1]
        eye = tf.eye(num_point // self.input_nfeats)
        one = tf.ones([batch_size, num_point  // self.input_nfeats])

        res = []
        print("Hello 1")
        for i in range(self.input_nfeats):
            print("Hello 2")
            res.append(
                tf.sort(
                    self.alpha[i] * tf.matmul(inputs[:, i::self.input_nfeats], eye)
                    + self.beta[i] * one
                )
            )
            print("Hello 3")
        return tf.concat(res, -1)



# Define NN model
with mirrored_strategy.scope():

    model=Sequential()

    model.add(Dense(10, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    # model.add(Dense(10))
    # model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.summary()

    #####################

    # inputs = keras.Input(shape=(n_feat_per_gal * n_mem_max,))
    # x = myLinear(input_nfeats=n_feat_per_gal)(inputs)
    # x = Activation('relu')(x)
    # outputs = Dense(3, activation='softmax')(x)
    # model2 = keras.Model(inputs, outputs)
    model2 = Sequential()
    model2.add(myLinear(input_nfeats=n_feat_per_gal))
    model2.add(Activation('relu'))
    model2.add(Dense(3, activation='softmax'))
    model2.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model2.summary()


# ANN hyperparamters
nb_epoch=300
batch_size= X_train.shape[0]

history = model.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    # shuffle=True,
    validation_data=(X_valid, Y_valid),
    # validation_batch_size=batch_size,
)

history2 = model2.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    # shuffle=True,
    validation_data=(X_valid, Y_valid),
    # validation_batch_size=batch_size,
)


class myLinear2(layers.Layer):
    def __init__(self,
                 input_nfeats,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(myLinear2, self).__init__(**kwargs)

        self.input_nfeats = input_nfeats

    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        alpha_init = tf.random_normal_initializer()
        beta_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(
            initial_value=alpha_init(
                shape=(self.input_nfeats,),
                dtype='float32',
            ),
            trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=beta_init(
                shape=(self.input_nfeats,),
                dtype='float32',
            ),
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        batch_size = inputs.get_shape()[0]
        num_point = inputs.get_shape()[1]
        eye = tf.eye(num_point // self.input_nfeats)
        one = tf.ones([batch_size, num_point  // self.input_nfeats])

        res = []
        for i in range(self.input_nfeats):
            res.append(
                tf.sort(
                    self.alpha[i] * tf.matmul(inputs[:, i::self.input_nfeats], eye)
                    + self.beta[i] * one
                )
            )
        return tf.concat(res, -1)
