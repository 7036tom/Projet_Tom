# coding=utf-8


import random
import csv
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad, SGD, Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge, Conv1D, BatchNormalization, PReLU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from scipy.stats import spearmanr, pearsonr
from scipy.stats.mstats import spearmanr as spearmanr_bis
import seaborn as sns
import matplotlib.pyplot as plt


sns.set()

model = load_model('models/model_tf_seed_1.h5') # Here enter the path to your model file

print(model.summary())

# Load and prepare the 25 songs long dataset from studyforest
studyframe= pd.read_csv("databases/Studyforestdb.csv", header=None)
studyset = studyframe.values
X = np.transpose(studyset.astype(float)) # It was stocked in a transposed format as is is faster to load this way

# Compute the fft of X
X_F = np.copy((X))
for i in range(25):
	X_F[i] = np.fft.fft(X[i], norm='ortho')

X= np.expand_dims(X, axis=2)
X_F = np.expand_dims(X_F, axis=2)



Conv1D1 = Model(inputs=model.input, outputs=model.get_layer('conv1d_1').output)
Conv1D2 = Model(inputs=model.input, outputs=model.get_layer('conv1d_2').output)

temp1 = Conv1D1.predict([X, X_F])
temp2 = Conv1D2.predict([X, X_F])

print(temp1.shape)

print(np.mean(temp1[0], axis=0).shape)

#Conv1D1_output = np.empty((25, 2*len(np.ravel(temp1[0])))) # Mean case
Conv1D1_output = np.empty((25, 2*48)) # This will contain the representation outputted by the first layer

for i in range(25):
	Conv1D1_output[i] = np.concatenate([np.mean(temp1[i], axis=0), np.mean(temp2[i],axis=0)])

#for i in range(25):
#	Conv1D1_output[i] = np.concatenate([np.ravel(temp1[i]), np.ravel(temp2[i])]) # Mean case

Conv1D3 = Model(inputs=model.input, outputs=model.get_layer('conv1d_3').output)
Conv1D4 = Model(inputs=model.input, outputs=model.get_layer('conv1d_4').output)

temp1 = Conv1D3.predict([X, X_F])
temp2 = Conv1D4.predict([X, X_F])

Conv1D2_output = np.empty((25, 2*128))
#Conv1D2_output = np.empty((25, 2*len(np.ravel(temp1[0])))) # Mean case

for i in range(25):
	Conv1D2_output[i] = np.concatenate([np.mean(temp1[i], axis=0), np.mean(temp2[i],axis=0)])

#for i in range(25):
#	Conv1D2_output[i] = np.concatenate([np.ravel(temp1[i]), np.ravel(temp2[i])]) # Mean case

Conv1D5 = Model(inputs=model.input, outputs=model.get_layer('conv1d_5').output)
Conv1D6 = Model(inputs=model.input, outputs=model.get_layer('conv1d_6').output)

temp1 = Conv1D5.predict([X, X_F])
temp2 = Conv1D6.predict([X, X_F])

Conv1D3_output = np.empty((25, 2*192))
#Conv1D3_output = np.empty((25, 2*len(np.ravel(temp1[0])))) # Mean case

for i in range(25):
	Conv1D3_output[i] =np.concatenate([np.mean(temp1[i], axis=0), np.mean(temp2[i],axis=0)])

#for i in range(25):
#	Conv1D3_output[i] =np.concatenate([np.ravel(temp1[i]), np.ravel(temp2[i])]) # Mean case


Conv1D7 = Model(inputs=model.input, outputs=model.get_layer('conv1d_7').output)
Conv1D8 = Model(inputs=model.input, outputs=model.get_layer('conv1d_8').output)

temp1 = Conv1D7.predict([X, X_F])
temp2 = Conv1D8.predict([X, X_F])

Conv1D4_output = np.empty((25, 2*192))
#Conv1D4_output = np.empty((25, 2*len(np.ravel(temp1[0])))) # Mean case

for i in range(25):
	Conv1D4_output[i] = np.concatenate([np.mean(temp1[i], axis=0), np.mean(temp2[i],axis=0)])

#for i in range(25):
#	Conv1D4_output[i] = np.concatenate([np.ravel(temp1[i]), np.ravel(temp2[i])]) # Mean case

Conv1D9 = Model(inputs=model.input, outputs=model.get_layer('conv1d_9').output)
Conv1D10 = Model(inputs=model.input, outputs=model.get_layer('conv1d_10').output)

temp1 = Conv1D9.predict([X, X_F])
temp2 = Conv1D10.predict([X, X_F])

Conv1D5_output = np.empty((25, 2*128))
#Conv1D5_output = np.empty((25, 2*len(np.ravel(temp1[0])))) # Mean case

for i in range(25):
	Conv1D5_output[i] = np.concatenate([np.mean(temp1[i], axis=0), np.mean(temp2[i],axis=0)])

#for i in range(25):
	#Conv1D5_output[i] = np.concatenate([np.ravel(temp1[i]), np.ravel(temp2[i])]) # Mean case


Dense1 = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
Dense1_output = Dense1.predict([X, X_F])

Dense2 = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
Dense2_output = Dense2.predict([X, X_F])


# DSM C1 #################################################################################################################################################################################

dsm1 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D1_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D1_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) # Metric choice for DNN representations comparison

		dsm1[i, j]=r#_mean


max = np.amax(dsm1)
min = np.amin(dsm1)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm1[i, j] = 1 - dsm1[i, j]
		#dsm[i, j] /= (1-min)

		#if i == j: # sometime, having zeros on the DSMs diag can mess up color contrasts (if for exemple or the other values are between 0.9 and 0.1). In this case, replacing these zeros by any values among the values of the DSM outside the diag allowd for better contrasts
		#	dsm1[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
# This can be useful to see if there are the clusters we want for ex

dsm_5x5_1 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_1[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm1[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_1[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_1[i, j]= dsm_5x5_1[j,i]


#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_1, delimiter=",")
np.savetxt("DSMs/L1_TF.csv", dsm1, delimiter=",") # Save the DSM of layer i so that it can be used with the rsa toolbox.

#ax = sns.heatmap(dsm.data, cmap="RdYlBu")

#plt.figure()
#ax = sns.heatmap(dsm_5x5_1, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm1, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99)) # Custom color palette, close to guclu's one
ax_bis.get_figure()



# DSM C2 ########################################################################################################################################################################"

dsm2 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Conv1D2_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Conv1D2_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		dsm2[i, j]=r#_mean


max = np.amax(dsm2)
min = np.amin(dsm2)
#print(max)


for i in range(25):
	for j in range(25):
		dsm2[i, j] = 1 - dsm2[i, j]
		if i == j:
			dsm2[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_2 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_2[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm2[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_2[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_2[i, j]= dsm_5x5_2[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_2, delimiter=",")
np.savetxt("DSMs/L2_TF.csv", dsm2, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_2, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm2, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))



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
		if i == j:
			dsm3[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_3 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_3[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm3[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_3[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_3[i, j]= dsm_5x5_3[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_3, delimiter=",")
np.savetxt("DSMs/L3_TF.csv", dsm3, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_3, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm3, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

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
		if i == j:
			dsm4[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_4 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_4[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm4[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_4[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_4[i, j]= dsm_5x5_4[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_4, delimiter=",")
np.savetxt("DSMs/L4_TF.csv", dsm4, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_4, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm4, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

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
		if i == j:
			dsm5[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_5 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_5[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm5[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_5[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_5[i, j]= dsm_5x5_5[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_5, delimiter=",")
np.savetxt("DSMs/L5_TF.csv", dsm5, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm5, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

# DSM C6 #############################################################################################################################################

dsm6 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Dense1_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Dense1_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm6[i, j]=r#_mean


max = np.amax(dsm6)
min = np.amin(dsm6)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm6[i, j] = 1 - dsm6[i, j]
		#dsm[i, j] /= (1-min)
		if i == j:
			dsm6[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_6 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_6[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm6[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_6[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_6[i, j]= dsm_5x5_6[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_6, delimiter=",")
np.savetxt("DSMs/L6_TF.csv", dsm6, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_6, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm6, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

# DSM C7 #############################################################################################################################################

dsm7 = np.empty((25,25))

# Using pearson r
activation = np.empty((25, len(Dense2_output[0]))) # Enter name of the layer you are interrested in
for i in range(25):
	activation[i]=np.copy(Dense2_output[i]) # Enter name of the layer you are interrested in

for i in range(25):
	for j in range(25):
		r, p = spearmanr(activation[i],activation[j]) #Pearson

		#print(r.shape)
		#print('r = ')
		#print(r)
		#r_mean = np.mean(r)
		dsm7[i, j]=r#_mean


max = np.amax(dsm7)
min = np.amin(dsm7)
#print(max)


for i in range(25):
	for j in range(25):
		#dsm[i, j] /= max
		dsm7[i, j] = 1 - dsm7[i, j]
		#dsm[i, j] /= (1-min)
		if i == j:
			dsm7[i,i] = 1-min


# Let's average cases about same music styles, so that we can compare the resulting DSM with brain DSM.
dsm_5x5_7 = np.empty(((5,5)))

for i in range(5):
	dsm_5x5_7[i,i] = 0

for i in range(4):
	for j in range(i,4):
		temp = np.average(dsm7[0 + i*5:5+i * 5, 5+j * 5:10 + j * 5])
		dsm_5x5_7[i,j+1]=temp

for i in range(5):
	for j in range(0,i):
		dsm_5x5_7[i, j]= dsm_5x5_7[j,i]
#for i in range(24):
#	dsm[i,i]=0

#np.savetxt("dsm_layers/Layer_7_TF_5X5.csv", dsm_5x5_7, delimiter=",")
np.savetxt("DSMs/L7_TF.csv", dsm7, delimiter=",")

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
#ax = sns.heatmap(dsm_5x5_7, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))
plt.figure()
ax_bis = sns.heatmap(dsm7, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))


# MODEL DSM (ie dsm of dsms!)

dsm_all = np.empty((7,7))

dsm = [dsm1, dsm2, dsm3, dsm4, dsm5, dsm6, dsm7]

for i in range(7):
	for j in range(7):
		r, p =spearmanr(np.ravel(dsm[i]),np.ravel(dsm[j]))
		dsm_all[i, j] = 1-r

plt.figure()
ax_bis = sns.heatmap(dsm_all, cmap=sns.diverging_palette(260, 10, sep=1, n=300, l=30, s=99.99))

plt.show()
