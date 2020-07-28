import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

fname = sys.argv[1]

log = np.genfromtxt('%s.log' % fname, names=True) 
plt.subplot(2,2,1) 
plt.plot(log['epoch'], log['loss'], label='loss') 
plt.plot(log['epoch'], log['val_loss'], label='val_loss') 
plt.legend() 
plt.xlabel('epoch') 
plt.subplot(2,2,2) 
plt.plot(log['epoch'], log['accuracy'], label='accuracy') 
plt.plot(log['epoch'], log['val_accuracy'], label='val_accuracy') 
plt.legend() 
plt.xlabel('epoch') 
plt.subplot(2,2,3) 
plt.plot(log['epoch'], log['loss'] - log['val_loss']) 
plt.legend() 
plt.xlabel('epoch') 
plt.subplot(2,2,4) 
plt.plot(log['epoch'], log['accuracy'] - log['val_accuracy']) 
plt.legend() 
plt.xlabel('epoch') 
plt.savefig('%s_metrics.pdf' % fname)

