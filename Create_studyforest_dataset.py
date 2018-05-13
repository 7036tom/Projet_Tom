import pandas as pd
import numpy as np
import librosa 
import csv
import random
import os
import sys
import time
from scipy.io.wavfile import read

Data = np.empty((25,96000))

No_file = 0 
for root, dirs, files, in os.walk("./raw_studyforest_data/"):
	for filename in sorted(files):
		X, sr = librosa.load('./raw_studyforest_data/'+filename, res_type='kaiser_fast', sr=16000)
		Data[No_file]= np.copy(X[0:96000])
		No_file=No_file+1


np.savetxt('./databases/Studyforestdb.csv', np.transpose(Data), delimiter=',')





