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
from theano import tensor as T
from sklearn.metrics import roc_auc_score, roc_curve

# fix random seed for reproducibility
seed = 0
np.random.seed(seed)

X = np.empty((1825,465984))

nb_line = 0
with open('databases/dbwhole_test.csv', 'rb') as f:
	while(nb_line < 1825):
		reader = csv.reader(f)
		row = reader.next()
		X[nb_line]=np.copy(row)
		nb_line = nb_line + 1

np.savetxt("databases/dbwhole1825.csv", np.transpose(X), delimiter=',')
