import os, sys
import numpy as np
import tensorflow as tf

# pathData="/mnt/local-scratch/u/ilic"
pathData="/home/users/ilic/ML/SDSS_image_data"
path_train = pathData + "/train"
path_valid = pathData + "/valid"

print("#####################################################")

# mirrored_strategy = tf.distribute.MirroredStrategy(
#     devices=[
#         "GPU:0",
#         "GPU:1",
#         # "GPU:2",
#     ]
# )
mirrored_strategy = tf.distribute.MirroredStrategy()

print("#####################################################")

# sys.exit()


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.normalization import  BatchNormalization
# from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import  BatchNormalization
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D


n_train = sum([len(os.listdir("%s/%s" % (path_train, d))) for d in os.listdir(path_train)])

batch_size = 16
target_size = (1024, 1024)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=target_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=target_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)

dropoutpar = 0.5
nb_dense = 64

with mirrored_strategy.scope():

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

steps_per_epoch = int(n_train  // batch_size)
validation_steps = int(n_train * 0.2 // batch_size)

# sys.exit()

# history = model.fit_generator(
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
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
