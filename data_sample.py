import pandas as pd
import random 
import numpy as np
import sys

filename = 'clean.csv'

samples = 1000

if len(sys.argv) > 1:
	samples = int(sys.argv[1])
	if len(sys.argv) > 2:
		filename = sys.argv[2]	
else: 
	print("Please spicify the size of the sample you want")
	print("Using default 1000")

print "Using: " +  str(samples) + ' samples'
train_test = pd.read_csv(filename, nrows = samples) 
train_test = train_test.sort_values(by = ['srch_id']) 

train = train_test[:int(0.9*len(train_test))]
test = train_test[int(-0.1*len(train_test)):]

train.to_csv('train'+'_'+str(samples)+'.csv')
test.to_csv('test'+'_'+str(samples)+'.csv')
