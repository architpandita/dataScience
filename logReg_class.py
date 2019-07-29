import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(z):
	h=1/(1+np.exp(-z))
	return h
	
def loss_fxn(h, y):
	return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

	

def training(xtrain,ytrain,rounds):
	m, n = xtrain.shape
	ytrain = ytrain.reshape(m, 1)
	theta = np.zeros((n + 1, 1))
	alpha = 0.05
	col_ones = np.ones((m, 1))
	xtrain = np.append(col_ones, xtrain, axis=1)
	total_loss = []
	for i in range(rounds):
		z = np.dot(xtrain, theta)
		h = sigmoid(z)
		gradient = np.dot(xtrain.T, (h - ytrain)) / m
		theta -= alpha*gradient
		total_loss.append(loss_fxn(h, ytrain))
	
	plt.plot(total_loss)
	plt.show()

df=pd.read_csv('processed.csv')
ydata=df['class'].values
df=df.drop(['class'],axis=1)
xdata=df.values

xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata,train_size=0.8)
training(xtrain, ytrain, 1000)


