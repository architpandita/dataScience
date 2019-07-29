import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def euclidean(x,xi,length):
	d=np.zeros(x.shape[0],1)
	d+=np.square(x-xi)
	
	return math.sqrt(d)
def knn(xtrain,xtest, k):
	distance={}
	sort={}
	length=xtest.shape[1]
	
	#step3 euclidean distance of each row of training and test test data
	for x in range(len(xtrain)):
		dis=euclidean(xtest, xtrain.iloc[x],length)
		distances[x]=dist[0]
	sorted_d=sorted(distances.items(),key=operator.itemgetter(1))
	neigh=[]
	
		
	
	
	

df=pd.read_csv('processed.csv')
ydata=df['class'].values
df=df.drop(['class'],axis=1)
xdata=df.values

xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata,train_size=0.8)
training(xtrain, ytrain, 10)
