# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

from tqdm import tqdm

from healpy.projector import GnomonicProj as GP

import os, sys

from sklearn import preprocessing
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import  BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.optimizers import rmsprop
import pdb
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score,auc

import matplotlib.pyplot as plt

#################################################################################

# Path to fits files
pathData="/home/silic/Downloads/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

# Some shuffling and thinning
thin = 1
ix = np.random.permutation(n_clu)
clu_full_data = clu_full_data[ix][::thin]

# Build training images
### Those just fits all the original clusters
xsize, reso = 64+1, 0.52
# xsize, reso = 128+1, 0.26
# xsize, reso = 256+1, 0.13
### To allow for shift
xmax_shift = 10
xsize += 2 * xmax_shift
### New images generation parameters
n_rand_rot = 4
n_rand_shift = 4
### Actual generation
X_imgs = np.zeros((n_clu * n_rand_rot * n_rand_shift, xsize, xsize, 1))
Y_imgs = np.zeros((n_clu * n_rand_rot * n_rand_shift, xsize, xsize))
ct = 0
for i in tqdm(range(n_clu)):
    g = mem_full_data['ID'] == clu_full_data['ID'][i]
    for j in range(n_rand_rot):
        rand_rot = np.random.randint(360)
        proj = GP(
            rot=[clu_full_data['RA'][i], clu_full_data['DEC'][i], rand_rot],
            xsize=xsize,
            reso=reso,
        )
        x, y = proj.ang2xy(
            mem_full_data['RA'][g],
            mem_full_data['DEC'][g],
            lonlat=True
        )
        ixs, jxs = proj.xy2ij(x, y)
        for k in range(n_rand_shift):
            sx = np.random.randint(-xmax_shift, xmax_shift + 1)
            sy = np.random.randint(-xmax_shift, xmax_shift + 1)
            for ix, jx in zip(ixs, jxs):
                X_imgs[ct, ix + sx, jx + sy] += 1.
            Y_imgs[ct, xsize // 2 + sx, xsize // 2 + sy] = 1.
            ct += 1

# Build label images
Y_imgs = Y_imgs.reshape(n_clu, -1)

# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training 
max_ix = int(n_clu * frac)
X_train = X_imgs[:max_ix,:, :, :]
X_test = X_imgs[max_ix:,:, :, :]
Y_train = Y_imgs[:max_ix, :]
Y_test = Y_imgs[max_ix:, :]


# Build CNN model
dropoutpar=0.5
nb_dense = 64
model=Sequential()

model.add(Conv2D(32, (6, 6), padding='same',input_shape=(xsize, xsize, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(nb_dense, activation='relu'))
model.add(Dropout(dropoutpar)) 

model.add(Dense(xsize*xsize, activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

sys.exit()

batch_size = 32 
nb_epoch = 20
data_augmentation = True

history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_test, Y_test),
)























'''
else:
  print('Using real-time data augmentation.')

  # this will do preprocessing and realtime data augmentation
  datagen = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False, 
            featurewise_std_normalization=False, 
            samplewise_std_normalization=False,
            zca_whitening=False, 
            rotation_range=25,
            width_shift_range=0.1,  
            height_shift_range=0.1, 
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=[0.75,1.3])  

        
  datagen.fit(x_train)
        
  history = model.fit_generator(
                    datagen.flow(x_train, t_train, batch_size=batch_size),
                    samples_per_epoch=x_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(x_test, t_test),
                    callbacks=[ earlystopping, modelcheckpoint,tensorboard]
                )



print("Saving model...")
model.save_weights(pathout+model_name+".hd5",overwrite=True)

"""## Predictions and comparisons of different approaches
The following cells use the trained models (RF,ANN and CNN) to predict the morphological class of the test dataset and compare the performance of the different algorithms.
"""

print("Predicting...")
print("====================")
model.load_weights(pathout+model_name+".hd5")

Y_pred_RF=clf.predict_proba(X_ML_test)[:,1]
print(Y_pred_RF.shape)
Y_pred_ANN=ann.predict(np.expand_dims(X_ML_test,2))
Y_pred_DL = model.predict(x_test)

"""We now compute the global accuracy as well as ROC and P-R curves. If you are not familiar with these curves please see the lecture slides or click [here](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)"""

#global accuracy

Y_pred_RF_class=Y_pred_RF*0
Y_pred_RF_class[Y_pred_RF>0.5]=1


Y_pred_ANN_class=Y_pred_ANN*0
Y_pred_ANN_class[Y_pred_ANN>0.5]=1

Y_pred_DL_class=Y_pred_DL*0
Y_pred_DL_class[Y_pred_DL>0.5]=1

print("Global Accuracy RF:", accuracy_score(Y_ML_test, Y_pred_RF_class))
print("Global Accuracy ANN:", accuracy_score(Y_ML_test, Y_pred_ANN_class))
print("Global Accuracy CNN:", accuracy_score(t_test, Y_pred_DL_class))




# ROC curve (False positive rate vs. True positive rate)
fpr_DL, tpr_DL, thresholds_DL = roc_curve(t_test, Y_pred_DL)
fpr_RF, tpr_RF, thresholds_RF = roc_curve(Y_ML_test, Y_pred_RF)
fpr_ANN, tpr_ANN, thresholds_ANN = roc_curve(Y_ML_test, Y_pred_ANN)

print("AUC RF:", auc(fpr_RF, tpr_RF))
print("AUC ANN:", auc(fpr_ANN, tpr_ANN))
print("AUC CNN:", auc(fpr_DL, tpr_DL))

#plot ROC
fig = plt.figure() 
title('ROC curve',fontsize=18)
xlabel("FPR", fontsize=20)
ylabel("TPR", fontsize=20)
xlim(0,1)
ylim(0,1)
plot(fpr_DL,tpr_DL,linewidth=3,color='red',label='CNN')
plot(fpr_RF,tpr_RF,linewidth=3,color='blue',label='RF')
plot(fpr_ANN,tpr_ANN,linewidth=3,color='green',label='ANN')
legend(fontsize=14)


# Precision Recall curve (False positive rate vs. True positive rate)
precision_DL, recall_DL, thresholds_DL = precision_recall_curve(t_test, Y_pred_DL)
precision_RF, recall_RF, thresholds_RF = precision_recall_curve(Y_ML_test, Y_pred_RF)
precision_ANN, recall_ANN, thresholds_ANN = precision_recall_curve(Y_ML_test, Y_pred_ANN)
#plot PR curve
fig = plt.figure() 
title('P-R curve',fontsize=18)
xlabel("Precision", fontsize=20)
ylabel("Recall", fontsize=20)
xlim(0,1)
ylim(0,1)
plot(precision_DL,recall_DL,linewidth=3,color='red',label='CNN')
plot(precision_RF,recall_RF,linewidth=3,color='blue',label='RF')
plot(precision_ANN,recall_ANN,linewidth=3,color='green',label='ANN')
legend(fontsize=14)

"""The follwing cells visualize some random examples of bad classifications in order to explore what the network has understood. If you run multiple times the examples will change.

### Bad classifications of CNNs
"""

# objects classifed as early-types by the CNN but visually classifed as late-types
bad = np.where((Y_pred_DL[:,0]<0.5)&(t_test==1))
randomized_inds_train = np.random.permutation(bad)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class1 but classified as Class0",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(x_test[i,:,:])
  plt.title('$Morph$='+str(t_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)



# objects classifed as late-types by the CNN but visually classifed as early-types
bad2 = np.where((Y_pred_DL[:,0]>0.5)&(t_test==0))
randomized_inds_train = np.random.permutation(bad2)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class0 but classified as Class1",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(x_test[i,:,:])
  plt.title('$Morph$='+str(t_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)

"""### Bad classifcations of RFs"""

# objects classifed as early-types by the RF but visually classifed as late-types
bad = np.where((Y_pred_RF<0.5)&(Y_ML_test==1))
randomized_inds_train = np.random.permutation(bad)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class1 but classified as Class0",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(I_ML_test[i,:,:])
  plt.title('$Morph$='+str(Y_ML_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)



# objects classifed as late-types by the CNN but visually classifed as early-types
bad2 = np.where((Y_pred_RF>0.5)&(Y_ML_test==0))
randomized_inds_train = np.random.permutation(bad2)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class0 but classified as Class1",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(I_ML_test[i,:,:])
  plt.title('$Morph$='+str(Y_ML_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)
  
#visualize the feature space
fig = plt.figure()
xlabel("$Log(M_*)$", fontsize=20)
ylabel("g-r", fontsize=20)
xlim(8,12)
ylim(0,1.2)
scatter(X_ML_test[bad[0],1],X_ML_test[bad[0],0],color='pink',s=25,label="S class. as E")
scatter(X_ML_test[bad2[0],1],X_ML_test[bad2[0],0],color='orange',s=25,label='E class. as S') 
legend(fontsize=14)

"""### Bad classifications of ANNs"""

# objects classifed as early-types by the ANN but visually classifed as late-types
bad = np.where((Y_pred_ANN[:,0]<0.5)&(Y_ML_test==1))
randomized_inds_train = np.random.permutation(bad)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class1 but classified as Class0",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(I_ML_test[i,:,:])
  plt.title('$Morph$='+str(Y_ML_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)



# objects classifed as late-types by the ANN but visually classifed as early-types
bad2 = np.where((Y_pred_ANN[:,0]>0.5)&(Y_ML_test==0))
randomized_inds_train = np.random.permutation(bad2)

fig = plt.figure()
fig.suptitle("Galaxies visually classifed as Class0 but classified as Class1",fontsize=10)
for i,j in zip(randomized_inds_train[0][0:4],range(4)):
  ax = fig.add_subplot(2, 2, j+1)
  im = ax.imshow(I_ML_test[i,:,:])
  plt.title('$Morph$='+str(Y_ML_test[i]))
  fig.tight_layout() 
  fig.colorbar(im)
  
#visualize the feature space
fig = plt.figure()
xlabel("$Log(M_*)$", fontsize=20)
ylabel("g-r", fontsize=20)
xlim(8,12)
ylim(0,1.2)
scatter(X_ML_test[bad[0],1],X_ML_test[bad[0],0],color='pink',s=25,label="S class. as E")
scatter(X_ML_test[bad2[0],1],X_ML_test[bad2[0],0],color='orange',s=25,label='E class. as S') 
legend(fontsize=14)
'''