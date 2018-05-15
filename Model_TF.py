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
seed = 3
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
X_test96K_F = np.expand_dims(np.fft.fft(X_test[:, 96000:2*96000], norm='ortho'), axis=2)

# 5329 items in test

# def generator

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
				#start = time.time()
				X[nb_elements_in_batch]=np.copy(row[Random[i]:Random[i]+96000])
				Y[nb_elements_in_batch]=np.copy(dataset_labels[int(float(row[len(row)-1]))].astype(int))
				#end = time.time()
				#print(end - start)
				nb_elements_in_batch +=1
			if (nb_elements_in_batch == 32): 
				
				nb_elements_in_batch = 0
				yield ([np.expand_dims(X, axis=2), np.expand_dims(np.fft.fft(X, norm='ortho'), axis=2)], Y) # 1e-5
					
generator = Generator()


@threadsafe_generator
def Generator_test():
	X = np.empty((32, 96000))
	Y = np.empty((32, 50))
	#nbr_samples = 464000
	# 100e-9

	# 387E-6

	nb_line = 0
	nb_cut = 24
	while 1:
		for i in range(nb_cut):
			X[i]=np.copy(X_test[nb_line, 16000*i:96000+16000*i])
			Y[i]=np.copy(dataset_labels[int(float(X_test[nb_line, len(X_test[0])-1]))])
		yield(np.expand_dims(X, axis=2), Y)
		nb_line+=1
		if nb_line == 1825:
			nb_line = 0


generator_test = Generator_test()






# Define network architecture

# def of the optimizer
ADAM = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-8) # Parameters from Guclu paper

# Parameters from Guclu paper

Input_T = Input(shape = (96000,1))
Input_F = Input(shape = (96000,1))

output_T_1 = BatchNormalization(input_shape = (96000,1))(Input_T)
output_F_1 = BatchNormalization(input_shape = (96000,1))(Input_F)

output_T_2 = Conv1D(filters=48 , kernel_size = 121, strides = 16,activation='relu')(output_T_1)
output_F_2 = Conv1D(filters=48 , kernel_size = 121, strides = 16,activation='relu')(output_F_1)

output_T_3 = MaxPooling1D(pool_size=9, strides=4)(output_T_2)
output_F_3 = MaxPooling1D(pool_size=9, strides=4)(output_F_2)

output_T_4 = Conv1D(filters=128, kernel_size = 25, activation='relu')(output_T_3)
output_F_4 = Conv1D(filters=128, kernel_size = 25, activation='relu')(output_F_3)

output_T_5 = MaxPooling1D(pool_size= 9, strides=4, padding='valid')(output_T_4)
output_F_5 = MaxPooling1D(pool_size= 9, strides=4, padding='valid')(output_F_4)

output_T_6 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_T_5)
output_F_6 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_F_5)

output_T_7 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_T_6)
output_F_7 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_F_6)

output_T_8 = Conv1D(filters=128, kernel_size = 9, activation='relu')(output_T_7)
output_F_8 = Conv1D(filters=128, kernel_size = 9, activation='relu')(output_F_7)

output_T_9 = MaxPooling1D(pool_size=9, strides=4, padding='valid')(output_T_8)
output_F_9 = MaxPooling1D(pool_size=9, strides=4, padding='valid')(output_F_8)

output_T = Flatten()(output_T_9)
output_F = Flatten()(output_F_9)

input_dense_part = merge([output_T, output_F], mode='concat')

output_dense_1 = Dense(4096, activation='relu')(input_dense_part)

output_dense_2 = Dropout(0.5)(output_dense_1)

output_dense_3 = Dense(4096, activation='relu')(output_dense_2)

output_dense_4 = Dropout(0.5)(output_dense_3)

output = Dense(50, activation='sigmoid')(output_dense_4)

model = Model([Input_T, Input_F], output)

model.compile(optimizer=ADAM, loss='binary_crossentropy', metrics=['accuracy'])





# Let's see what the AUC is doing during training
class MonitorAUC_train(Callback):
    def on_epoch_end(self, epoch, logs={}):
        yhat = self.model.predict([X_test96K, X_test96K_F], verbose=0)
        print ' AUC :', roc_auc_score(Y_test, yhat)

callbacks = [
   	MonitorAUC_train(),
	EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
	ModelCheckpoint('weights/weight_tf_seed3.hdf5', monitor='val_loss', verbose=0, save_best_only='true', save_weights_only='true', mode='auto', period=1)
]


# 625
model.fit_generator(generator,validation_data=([X_test96K, X_test96K_F], Y_test), callbacks=callbacks, verbose = 1, epochs=200, max_queue_size=100, workers=1, steps_per_epoch=200, use_multiprocessing='false')

model.save('models/model_tf_seed3.hdf5.h5')




