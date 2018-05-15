# coding=utf-8
import csv
import numpy as np
import pandas as pd

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
