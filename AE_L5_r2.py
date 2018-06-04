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
from keras.layers import UpSampling1D,Dense, Reshape, Activation, Dropout,Layer, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge, Conv1D, BatchNormalization, PReLU, LocallyConnected1D
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

# Definition of test data
dataframe_audio_test = pd.read_csv("databases/dbwhole1825.csv", header=None)
dataset_audio_test = dataframe_audio_test.values
X_test = np.transpose(dataset_audio_test.astype(float)) # X_test is data to best prepared with generator_test

X_test96K = np.copy(X_test[:, 96000:2*96000])

# Preprocessing

X_test96K = librosa.util.normalize(X_test96K, axis=1) # Put data between -1 and 1


# def generator (cf details in section Model_TF)

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
	X = np.empty((16, 96000))
	i = 0

	#88e9
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
		
				

			
			 
			if (nb_elements_in_batch < 16):
				#start = time.time()
				X[nb_elements_in_batch]=np.copy(row[Random[i]:Random[i]+96000])
				#end = time.time()
				#print(end - start)
				nb_elements_in_batch +=1
				i = i + 1
			if (nb_elements_in_batch == 16):

				# Preprocessing

				X = librosa.util.normalize(X, axis=1)
				
				nb_elements_in_batch = 0
				yield (np.expand_dims(X, axis=2), np.expand_dims(X, axis=2)) # 1e-5 # Contrary to Model TF case, here 'Y = X' as we are are talking about an AE
					
generator = Generator()








# Define network architecture

# def of the optimizer
ADAM = Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=1e-8) # Parameters from Guclu paper
# Parameters from Guclu paper

"""
Input_F = Input(shape = (96000,1))
output_F_1 = BatchNormalization(input_shape = (96000,1))(Input_F)
output_F_2 = Conv1D(filters=48 , kernel_size = 121, strides = 16)(output_F_1)
output_F_3 = MaxPooling1D(pool_size=9, strides=4)(output_F_2)
output_F_4 = Conv1D(filters=128, kernel_size = 25, activation='relu')(output_F_3)
output_F_5 = MaxPooling1D(pool_size= 9, strides=4, padding='same')(output_F_4)
output_F_6 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_F_5)
output_F_7 = Conv1D(filters=192, kernel_size = 9, activation='relu')(output_F_6)
output_F_8 = Conv1D(filters=128, kernel_size = 9, activation='relu')(output_F_7)
output_F_9 = MaxPooling1D(pool_size=9, strides=4, padding='same')(output_F_8)
output_F = Flatten()(output_F_9)
"""

Input_T = Input(shape = (96000,1))
output_T_1 = BatchNormalization()(Input_T)

output_T_2 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(output_T_1)

output_T_2 = MaxPooling1D(pool_size=4, padding='same')(output_T_2)

output_T_3 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(output_T_2)

output_T_3 = MaxPooling1D(pool_size=4, padding='same')(output_T_3)

output_T_4 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(output_T_3)

output_T_4 = MaxPooling1D(pool_size=2, padding='same')(output_T_4)

output_T_5 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(output_T_4)

output_T_5 = MaxPooling1D(pool_size=2, padding='same')(output_T_5)

output_T_6 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(output_T_5)




Encoded = MaxPooling1D(pool_size=2 ,padding='same')(output_T_6)




decoded1 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(Encoded)

decoded1 = UpSampling1D(size=2)(decoded1)

decoded1 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(decoded1)

decoded1 = UpSampling1D(size=2)(decoded1)

decoded1 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(decoded1)

decoded1 = UpSampling1D(size=2)(decoded1)

decoded1 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(decoded1)

decoded1 = UpSampling1D(size=4)(decoded1)

decoded1 = Conv1D(filters=60 , kernel_size = 16,activation='relu', padding='same')(decoded1)

decoded1 = UpSampling1D(size=4)(decoded1)

decoded = Conv1D(filters=1, kernel_size=16, activation = 'tanh', padding='same')(decoded1) # tanh because the value of X are normalized between -1 and 1

Autoencodeur = Model(Input_T, decoded)
Autoencodeur.compile(optimizer='Adadelta', loss='mean_squared_error', metrics=['mean_squared_error'])
# CHANGE : ADADELTA
print(Autoencodeur.summary())

X_test96K = np.expand_dims(X_test96K, axis=2) # This one is for use in AUC_callback

callbacks = [
	EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
	ModelCheckpoint('weights/weight_autoenc_5L_128.hdf5', monitor='val_loss', verbose=0, save_best_only='true', save_weights_only='true', mode='auto', period=1)
]

Autoencodeur.fit_generator(generator,validation_data=(X_test96K, X_test96K), callbacks=callbacks, verbose = 1, epochs=200, max_queue_size=100, workers=1, steps_per_epoch=400, use_multiprocessing='false')

Autoencodeur.save('models/Auto_encoder_5L_128.h5')

pred = Autoencodeur.predict(X_test96K)


#Autoencodeur.save('model_autoencodeur.h5')




