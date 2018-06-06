# coding=utf-8

import sys
import random
import csv
import numpy as np
import pandas as pd
import math
import threading
from keras.models import load_model


from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad, SGD, Adam
from keras.models import Sequential, Model
from keras.layers import UpSampling1D, Dense, Reshape, Activation, Dropout,Layer, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge, Conv1D, BatchNormalization, PReLU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from scipy.stats import spearmanr

import librosa
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sns.set()

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




Encoded = MaxPooling1D(pool_size=2, padding='same')(output_T_6)




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

decoded = Conv1D(filters=1, kernel_size=16, activation = 'tanh', padding='same')(decoded1)

Autoencodeur = Model(Input_T, decoded)
Autoencodeur.compile(optimizer='Adadelta', loss='mean_squared_error', metrics=['mean_squared_error'])
# CHANGE : ADADELTA
print(Autoencodeur.summary())

Autoencodeur.load_weights('../models/weight_autoenc_5L_2.13.hdf5')
"""

Autoencodeur = load_model('models/Auto_encoder_5L_10_truediff2.0.h5')


studyframe= pd.read_csv("databases/Studyforestdb.csv", header=None)
studyset = studyframe.values
X = np.transpose(studyset.astype(float))

X = librosa.util.normalize(X, axis=1)



X= np.expand_dims(X, axis=2)

Conv1D1 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_1').output)
Conv1D1_output = np.mean(Conv1D1.predict(X), axis=1) # Contralement à DSM_TF.py, les modèles que nous avons étudiés ici sont des autoencodeurs ne considerant que la representation temporelle de l'entrée : donc pas de concatenation de deux representations.

print(Conv1D1.predict(X).shape)

Conv1D2 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_2').output)
Conv1D2_output = np.mean(Conv1D2.predict(X), axis=1)

Conv1D3 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_3').output)
Conv1D3_output = np.mean(Conv1D3.predict(X), axis=1)

Conv1D4 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_4').output)
Conv1D4_output = np.mean(Conv1D4.predict(X), axis=1)

Conv1D5 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_5').output)
Conv1D5_output = np.mean(Conv1D5.predict(X), axis=1)

Conv1D6 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_6').output)
Conv1D6_output = np.mean(Conv1D6.predict(X), axis=1)

Conv1D7 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_7').output)
Conv1D7_output = np.mean(Conv1D7.predict(X), axis=1)

Conv1D8 = Model(inputs=Autoencodeur.input, outputs=Autoencodeur.get_layer('conv1d_8').output)
Conv1D8_output = np.mean(Conv1D8.predict(X), axis=1)


# DSM C1 #################################################################################################################################################################################

dsm1 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D1_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D1_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm1[i, j]=r#_mean


max = np.amax(dsm1)
min = np.amin(dsm1)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm1[i, j] = 1 - dsm1[i, j]
		#dsm[i, j] /= (1-min)
		#if i == j:
			#dsm1[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm1[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5[i, j]= dsm_5x5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5, delimiter=",")
np.savetxt("DSMs/L1_AE.csv", dsm1, delimiter=",")

# Using a random norm
"""
for i in range(24):
	for j in range(24):
		dsm[i,j]=np.linalg.norm(Conv1D4_output[i]-Conv1D4_output[j])

#row_sums = dsm.sum(axis=1)
#dsm_norm = dsm / row_sums[:, np.newaxis]

DSM = DissimilarityMatrix(dsm) # Conversion from symilarity to dissymilarity
"""


#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

#plt.figure()
#ax = sns.heatmap(dsm_5x5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

ax_bis = sns.heatmap(dsm1, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
fig = ax_bis.get_figure()
fig.savefig("DSMs_png/L1_AE.png")

# DSM C2 ########################################################################################################################################################################"

dsm2 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D2_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D2_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm2[i, j]=r#_mean


max = np.amax(dsm2)
min = np.amin(dsm2)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm2[i, j] = 1 - dsm2[i, j]
		#dsm[i, j] /= (1-min)
		#if i == j:
		#	dsm2[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm2[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5[i, j]= dsm_5x5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5, delimiter=",")
np.savetxt("DSMs/L2_AE.csv", dsm2, delimiter=",")

# Using a random norm
"""
for i in range(24):
	for j in range(24):
		dsm[i,j]=np.linalg.norm(Conv1D4_output[i]-Conv1D4_output[j])

#row_sums = dsm.sum(axis=1)
#dsm_norm = dsm / row_sums[:, np.newaxis]

DSM = DissimilarityMatrix(dsm) # Conversion from symilarity to dissymilarity
"""


#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

#plt.figure()
#ax = sns.heatmap(dsm_5x5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
ax_bis = sns.heatmap(dsm2, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
fig = ax_bis.get_figure()
fig.savefig("DSMs_png/L2_AE.png")



# DSM C3 #############################################################################################################################################

dsm3 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D3_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D3_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm3[i, j]=r#_mean


max = np.amax(dsm3)
min = np.amin(dsm3)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm3[i, j] = 1 - dsm3[i, j]
		#dsm[i, j] /= (1-min)
		#if i == j:
		#	dsm3[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm3[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5[i, j]= dsm_5x5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5, delimiter=",")
np.savetxt("DSMs/L3_AE.csv", dsm3, delimiter=",")


# Using a random norm
"""
for i in range(24):
	for j in range(24):
		dsm[i,j]=np.linalg.norm(Conv1D4_output[i]-Conv1D4_output[j])

#row_sums = dsm.sum(axis=1)
#dsm_norm = dsm / row_sums[:, np.newaxis]

DSM = DissimilarityMatrix(dsm) # Conversion from symilarity to dissymilarity
"""


#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

#plt.figure()
#ax = sns.heatmap(dsm_5x5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

ax_bis = sns.heatmap(dsm3, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
fig = ax_bis.get_figure()
fig.savefig("DSMs_png/L3_AE.png")

# DSM C4 #############################################################################################################################################

dsm4 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D4_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D4_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm4[i, j]=r#_mean


max = np.amax(dsm4)
min = np.amin(dsm4)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm4[i, j] = 1 - dsm4[i, j]
		#dsm[i, j] /= (1-min)
		#if i == j:
		#	dsm4[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm4[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5[i, j]= dsm_5x5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5, delimiter=",")
np.savetxt("DSMs/L4_AE.csv", dsm4, delimiter=",")

# Using a random norm
"""
for i in range(24):
	for j in range(24):
		dsm[i,j]=np.linalg.norm(Conv1D4_output[i]-Conv1D4_output[j])

#row_sums = dsm.sum(axis=1)
#dsm_norm = dsm / row_sums[:, np.newaxis]

DSM = DissimilarityMatrix(dsm) # Conversion from symilarity to dissymilarity
"""


#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

#plt.figure()
#ax = sns.heatmap(dsm_5x5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
ax_bis = sns.heatmap(dsm4, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
fig = ax_bis.get_figure()
fig.savefig("DSMs_png/L4_AE.png")

# DSM C5 #############################################################################################################################################

dsm5 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D5_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D5_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm5[i, j]=r#_mean


max = np.amax(dsm5)
min = np.amin(dsm5)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm5[i, j] = 1 - dsm5[i, j]
		#dsm[i, j] /= (1-min)
		#if i == j:
		#	dsm5[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm5[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5[i, j]= dsm_5x5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5, delimiter=",")
np.savetxt("DSMs/L5_AE.csv", dsm5, delimiter=",")

# Using a random norm
"""
for i in range(24):
	for j in range(24):
		dsm[i,j]=np.linalg.norm(Conv1D4_output[i]-Conv1D4_output[j])

#row_sums = dsm.sum(axis=1)
#dsm_norm = dsm / row_sums[:, np.newaxis]

DSM = DissimilarityMatrix(dsm) # Conversion from symilarity to dissymilarity
"""

#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

"""
results = Autoencodeur.predict(X)
print(np.mean(results[0]))
print(np.std(results[0]))

for i in range(25):
	librosa.output.write_wav('/home/tom/Documents/Reco/Reco5_advanced/true'+str(i), results[i], 16000, norm=False)
"""

#plt.figure()
#ax = sns.heatmap(dsm_5x5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
ax_bis = sns.heatmap(dsm5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
fig = ax_bis.get_figure()
fig.savefig("DSMs_png/L5_AE.png")


#np.savetxt("L1_AE_5L_64.csv", dsm1, delimiter=",")
#np.savetxt("L2_AE_5L_64.csv", dsm2, delimiter=",")
#np.savetxt("L3_AE_5L_64.csv", dsm3, delimiter=",")
#np.savetxt("L4_AE_5L_64.csv", dsm4, delimiter=",")
#np.savetxt("L5_AE_5L_64.csv", dsm5, delimiter=",")

plt.show()
