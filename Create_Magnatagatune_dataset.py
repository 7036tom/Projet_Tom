import pandas as pd
import numpy as np
import librosa 
import csv
import random
import sys
import time

# Extraction of the 50 most used labels and paths

df_sounds=pd.read_csv("raw_magnatagatune_data/annotations_final.csv", header=None, sep='\t')

dataset_sound= df_sounds.values

Labels = dataset_sound[1:len(dataset_sound),1:51].astype(float)   # We dont want header / we convert to float to be able to use it later.
Path = dataset_sound[1:len(dataset_sound),189:190].astype(str)     # We dont want the header / Here we need to keep the paths as strings as it is the requiered format.


k = 0
for i in range(1,len(dataset_sound[0])-1): # =189 (We neither want to look at the id column nor at the path column)
	som = 0
	for s in range(1, len(dataset_sound)): # We dont care about the header
		som += int(dataset_sound[s, i]) 
	if (som > 475):
		for j in range(1,len(dataset_sound)):
			Labels[j-1,k] = dataset_sound[j,i]
		k = k + 1

np.savetxt('databases/labels_Mtatune.csv', Labels, delimiter=',')

for j in range(1,len(dataset_sound)): # No header
	Path[j-1] = dataset_sound[j,len(dataset_sound[0])-1] # We load the last column, which corresponds to the path.




Taille_sousB = len(Path)

with open("databases/dbwhole.csv", "wb") as db:
	writer = csv.writer(db)
	for i in range(Taille_sousB): # taillesousb
		try:
			X, sample_rate = librosa.load("raw_magnatagatune_data/"+Path[i][0], res_type='kaiser_fast', sr = 16000)#, sr = 16000)  # Probably an error due to try : if someting fails X stays empty at the line of the failed event.                        ed error line"+str(i))
			X[len(X) - 1] = i
			writer.writerow(X)
		except:
			print("unexpected error line" + str(i))
