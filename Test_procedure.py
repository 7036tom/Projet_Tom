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
from keras.models import load_model

# Load model : insert path to model here
model = load_model('models/model_tf_seed_10.h5')

# Load labels
dataframe_labels = pd.read_csv("databases/labels_Mtatune.csv", header=None)
dataset_labels = dataframe_labels.values

# Creation of Preds and True
len_test = 4034 # nb elements in the test db
Preds = np.empty((len_test, 50))
True = np.empty((len_test, 50))

X_temp = np.empty((24, 96000))
Preds_temp = np.empty((24, 50))

# We can't just load and predict the whole test dataset as it is too memory intensive
with open('databases/dbwhole_test_true.csv', 'rb') as f:
	reader = csv.reader(f)
	nb_line = 0

	while(nb_line < len_test):
		row = reader.next()
		#print(nb_line)
		for i in range(24):
			X_temp[i]=np.copy(row[i*16000:i*16000+96000])
                print(X_temp.shape)
		Preds_temp = model.predict([np.expand_dims(X_temp,axis=2),np.expand_dims(np.fft.fft(X_temp, norm='ortho'), axis=2)], verbose=0, batch_size=12)
                # np.expand_dims(X_temp, axis=2)
		# We sum the 24 predictions
		for i in range(24):
			for j in range(50):
				Preds[nb_line, j]=Preds[nb_line, j]+Preds_temp[i,j]

		# We then average them
		for i in range(24):
			for j in range(50):
				Preds[nb_line, j] = Preds[nb_line, j]/24

		# Finally we define True
		True[nb_line]=np.copy(dataset_labels[int(float(row[len(row)-1]))].astype(int))
		nb_line = nb_line + 1

# We evaluate the quality of the prediction
r_score = roc_auc_score(True, Preds, average='macro')

print("AUC : " + str(r_score))


Preds[Preds>=0.5] = 1
Preds[Preds<0.5] = 0



for i in range(20):
	print(Preds[i])
	print(" \n")
	print(True[i])
	print(" -------------------- \n")


