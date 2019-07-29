import numpy as np
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split

def loss_fxn(h,ytrain):
	return sqrt((np.sum(ytrain-h)**2).mean())

def f(x):
	x=x[0]
	if x>0.5:
		return [1]
	else:
		return [0]

def linear(theta,xtrain,f=f):
		
	h=np.dot(xtrain,theta)
	return np.array(list(map(f,h)))
		
	


def train(xtrain,ytrain,rounds):
	m, n = xtrain.shape
	ytrain = ytrain.reshape(m, 1)
	#print(xtrain.shape)
	#print(ytrain.shape)
	alpha = 0.008
	col_ones = np.ones((m, 1))
	xtrain = np.append(col_ones, xtrain, axis=1)
	#print(xtrain)
	theta=np.random.uniform(0,1,n+1)
	
	theta=theta.reshape(n+1,1)
	
	total_loss = []
	
	
	for i in range(rounds):
		
		h=linear(theta,xtrain) 
		
		#print(h.shape, xtrain.shape, ytrain.shape)
		gradient = np.dot(xtrain.T, (h - ytrain)) / m
		
		#print(gradient.shape)
		#print(loss_fxn())
		theta-=alpha*gradient
		loss=loss_fxn(h,ytrain)
		
	return theta
		

def predict(theta,xtest):
	col_ones = np.ones((xtest.shape[0], 1))
	xtest = np.append(col_ones, xtest, axis=1)
	return linear(theta,xtest)

def accuracy(ypred,ytest):
	ytest=ytest.reshape(ytest.shape[0],1)
	correct=np.sum(ypred == ytest)		
	print(correct)
	print(ypred,len(ytest))
	
	print ("accuracy: ", np.mean((correct)*100/len(ytest)))	


df=pd.read_csv('processed.csv')
ydata=df['class'].values
df=df.drop(['class'],axis=1)
xdata=df.values

xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata,train_size=0.8)
theta=train(xtrain, ytrain, 500)
ypred=predict(theta,xtest)
accuracy(ypred,ytest)


