# coding=utf-8

import sys
import random
import csv
import numpy as np
import pandas as pd
import math
import librosa
import threading
import librosa

from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad, SGD, Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge, Conv1D, BatchNormalization, PReLU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from sklearn.metrics import roc_auc_score, roc_curve

# fix random seed for reproducibility
seed = 0
np.random.seed(seed)


# Generator definition
nbr_random = 1500000 # Should be more than a million in real situation
Random = np.zeros(nbr_random).astype(int) 
for i in range(nbr_random):
	Random[i]=int(random.random()*(465983-96001))

dataframe_labels = pd.read_csv("databases/labels_Mtatune.csv", header=None)
dataset_labels = dataframe_labels.values

# Definition of test data
dataframe_audio_test = pd.read_csv("databases/dbwhole1825.csv", header=None)
dataset_audio_test = dataframe_audio_test.values
X_test = np.transpose(dataset_audio_test.astype(float)) # X_test is data to best prepared with generator_test
Y_test = np.empty((1825, 50))
for i in range(1825):
	Y_test[i]=np.copy(dataset_labels[int(float(X_test[i, len(X_test[0])-1]))])

X_test96K = np.expand_dims(X_test[:, 96000:2*96000], axis=2) # This one is for use in AUC_callback

# 5329 items in test

# def generator

# This allows for a multithreading generator, however it seems multhithreading and callbacks don't work well together so 
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator 
def Generator():
	nb_elements_in_batch = 0
	size_of_one_line = 43.11e6
	file_size = 88e9
	X = np.empty((32, 96000))
	Y = np.empty((32, 50))

	#100e-9
	#387E-6
	
	with open('databases/dbwhole.csv', 'rb') as f:
		reader = csv.reader(f)

		while 1:
			offset = random.randrange(file_size-size_of_one_line)
			f.seek(offset)	# go to random position
			row = reader.next()	# discard - bound to be partial line

			# extra to handle last/first line edge cases
			if len(row) == 0:       # we have hit the end
				f.seek(0)
				row = reader.next()  # so we'll grab the first line instead

			row = reader.next()				# bingo!
			# extra to handle last/first line edge cases
			if len(row) == 0:       # we have hit the end
				f.seek(0)
				row = reader.next()  # so we'll grab the first line instead
		
				

			
			 
			if (nb_elements_in_batch < 32):
				X[nb_elements_in_batch]=np.copy(row[Random[i]:Random[i]+96000])
				Y[nb_elements_in_batch]=np.copy(dataset_labels[int(float(row[len(row)-1]))].astype(int))
				nb_elements_in_batch +=1
			if (nb_elements_in_batch == 32): 
				
				nb_elements_in_batch = 0
				yield(np.expand_dims(X, axis=2), Y) # 1e-5
					
generator = Generator()

# Define network architecture

# def of the optimizer
ADAM = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-8) # Parameters from Guclu paper

model = Sequential()
# Parameters from Guclu paper
# Input layer (Conv1)

model.add(BatchNormalization(input_shape = (96000,1)))

model.add(Conv1D(filters=48 , kernel_size = 121, strides = 16, activation='relu'))
# pool1
model.add(MaxPooling1D(pool_size=9, strides=4))

#model.add(BatchNormalization())
# Hidden layers
# Conv2
model.add(Conv1D(filters=128, kernel_size = 25, activation='relu'))#, strides = 4))

# Pool2
model.add(MaxPooling1D(pool_size= 9, strides=4, padding='valid'))

# Conv3
model.add(Conv1D(filters=192, kernel_size = 9, activation='relu'))

# Conv4
model.add(Conv1D(filters=192, kernel_size = 9, activation='relu'))

# Conv5
model.add(Conv1D(filters=128, kernel_size = 9, activation='relu'))

# Pool5
model.add(MaxPooling1D(pool_size=9, strides=4, padding='valid'))


model.add(Flatten())
# Fc6
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
# Fc7
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
# Output layer (fc8)
model.add(Dense(50, activation='sigmoid'))

model.compile(optimizer=ADAM, loss='binary_crossentropy', metrics=['accuracy'])


# Let's see what the AUC is doing during training
class MonitorAUC_train(Callback):
    def on_epoch_end(self, epoch, logs={}):
        yhat = self.model.predict(X_test96K, verbose=0)
        print ' AUC :', roc_auc_score(Y_test, yhat)

callbacks = [
   	MonitorAUC_train(),
	EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
	ModelCheckpoint('weights/weight.hdf5', monitor='val_loss', verbose=0, save_best_only='true', save_weights_only='true', mode='auto', period=1)
]


# 625
model.fit_generator(generator,validation_data=(X_test96K, Y_test), callbacks=callbacks, verbose = 1, epochs=200, max_queue_size=100, workers=1, steps_per_epoch=200, use_multiprocessing='false')

model.save('models/model_T.h5')




