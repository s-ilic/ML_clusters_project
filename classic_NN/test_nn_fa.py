import numpy as np
from tqdm import tqdm

import os, sys
import time
import urllib.request, json
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant

import matplotlib.pyplot as plt

# mirrored_strategy = tf.distribute.MirroredStrategy()

curr_time = int(time.time())
min_time = curr_time - 10*86400
max_time = curr_time + 10000

url_name = "https://api.dex.vision/chart/history?symbol=UniswapV2%3AFARMUSDC-0x514906FC121c7878424a5C928cad1852CC545892&resolution=1&from="
url_name += str(min_time)
url_name += "&to="
url_name += str(max_time)
with urllib.request.urlopen(url_name) as url:
    pdata = json.loads(url.read().decode())

n_samp = 30
allX = []
allY = []
index = np.arange(len(pdata['t']) - n_samp - 1)
np.random.shuffle(index)
for i in index:
    tmpX = np.zeros((n_samp, 5))
    tmpY = np.zeros(5)
    for ix, k in enumerate(['c', 'o', 'h', 'l', 'v']):
        tmpX[:, ix] = pdata[k][i:i+n_samp]
        tmpY[ix] = pdata[k][i+n_samp]
    allX.append(tmpX.flatten())
    allY.append(tmpY)
allX = np.array(allX)
allY = np.array(allY)
me = allX.mean(axis=(0, 1))
st = allX.std(axis=(0, 1))
allX = (allX - me) / st
allY = (allY - me) / st


# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(len(allX) * frac)
X_train = allX[:max_ix, ...]
X_test = allX[max_ix:, ...]
Y_train = allY[:max_ix, :]
Y_test = allY[max_ix:, :]

# ANN hyperparamters
nb_epoch=50000
batch_size= len(X_train)

# with mirrored_strategy.scope():
'''
inputs = tf.keras.Input(shape=allX.shape[1:])
# lay1 = GRU(
# lay1 = LSTM(
lay1 = Dense(
    100,
    activation="relu",
    # activation="tanh",
)
out_lay1 = lay1(inputs)
lay1b = Dropout(0.1)
lay2 = Dense(
    5,
    activation="linear",
)
output = lay2(lay1b(out_lay1))
model = tf.keras.Model(inputs, output)
model.compile(
    # loss='mean_absolute_percentage_error',
    loss='mean_squared_error',
    optimizer=Adam(
        learning_rate=0.001,
    ),
    metrics=['accuracy'],
)
model.summary()
model.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
)
'''

# NN with XX hidden layer of XX neurons
ann = Sequential()
ann.add(Dense(100, input_dim=(None, 150,))
# ann.add(LSTM(100, activation='relu'))
ann.add(Activation('relu'))
ann.add(Dense(5))
ann.add(Activation('linear'))
ann.compile(
    loss='mean_squared_error',
    # optimizer=Adam(),
    optimizer=Adam(lr=0.01),
    metrics=['accuracy']
)
# sys.exit()
# ann.summary()
ann.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
)

sys.exit()

#########################################################################

score = ann.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Y_test_from_ann = ann.predict(X_test)
plt.subplot(1,2,1)
plt.plot(Y_test[:, 0], Y_test_from_ann[:, 0], '+')
plt.plot([Y_test[:, 0].min(), Y_test[:, 0].max()], [Y_test[:, 0].min(), Y_test[:, 0].max()])

Y_train_from_ann = ann.predict(X_train)
plt.subplot(1,2,2)
plt.plot(Y_train[:, 0], Y_train_from_ann[:, 0], '+')
plt.plot([Y_train[:, 0].min(), Y_train[:, 0].max()], [Y_train[:, 0].min(), Y_train[:, 0].max()])

plt.savefig('z_phot_spec.pdf')
plt.show()

#########################################################################

fname = 'pofz-12-008158.dat'


# full_id = np.zeros(0, dtype=np.int64)
# full_pz = np.zeros((0, 35))

full_id = []
full_pz = []

for fname in tqdm(os.listdir('.')):
    if fname.startswith('pofz') and fname.endswith('.dat'):
        t = np.loadtxt(
            fname,
            dtype=np.dtype(
                [
                    ('objID', np.int64),
                    ('run', np.int32),
                    ('rerun', np.str),
                    ('camcol', np.int32),
                    ('field', np.int32),
                    ('id', np.int32),
                    ('ra', np.float64),
                    ('dec', np.float64),
                    ('cmodelmag_r', np.float32),
                    ('modelmag_umg', np.float32),
                    ('modelmag_gmr', np.float32),
                    ('modelmag_rmi', np.float32),
                    ('modelmag_imz', np.float32),
                ] +
                [
                    ('pofz_%s' % i, np.float32) for i in range(35)
                ]
            )
        )
        full_id = full_id.append(np.atleast_1d(t['objID']))
        try:
            tmp = np.zeros((len(t), 35))
        except:
            tmp = np.zeros((1, 35))
        for i in range(35):
            tmp[:, i] = t['pofz_%s' % i]
        full_pz.append(tmp.copy())

