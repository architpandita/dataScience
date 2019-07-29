# linear svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
def fit(data):
	#|w|: {w,b}
	opt_dict={}
	transforms=[[1,1],[-1,1],[-1,-1],[1,-1]]
	all_data=np.array([])
	
	for yi in data:
		all_data=np.append(all_data, data[yi])
	max_feature_value= max(all_data)
	min_feature_value= min(all_data)
	all_data=None
	
	#with smaller steps for precision
	step_size=[max_feature_value * 0.1, 
		max_feature_value * 0.01,
		max_feature_value * 0.001,]
	#extreme expense
	b_range_multiple=5
	b_multiple=5
	latest_optimum=max_feature_value*10
	#satisfy yi(x.w)+b>=1 	
	m=0
	for step in step_size:
		w=np.array([latest_optimum,latest_optimum])
		#due to convex
		optimized=False
		while not optimized:
			for b in np.arange(-1*max_feature_value*b_range_multiple,max_feature_value*b_range_multiple,
								step*b_multiple ):
				for transformation in transforms:
					w_t=  w*transformation
					found_option=True
					#weakest link in SVM fundamental
					#SMO attempt to fix 
					#ti(xi.w+b)> =1
					k=False
					for i in data:
						
						
						for xi in data[i][1:]:
							
							yi=i
							
							try:
								if not yi*(np.dot(w_t, xi)+b)>=1:
									found_option=False
							except:
								pass
					#if m>10:
					#	found_option=False
					m+=1	
					if found_option:
						#all pts satify y(w.x)+b>=1	for w_t,b
						#put w,b in dict with ||w|| as key
						opt_dict[np.linalg.norm(w_t)]=[w_t,b]
						
						#after w[0] or w[1]<0 , w start repeating bcoz tranformation
						print(w,len(opt_dict))
						if w[0]<0:
							optimized= True
							print("optimized a step")
						else:
							w=w-step	
						#sorting \w\
						norms=sorted([n for n in opt_dict])
						#optimal val w,b
						opt_choice= opt_dict[norms[0]]
						w=opt_choice[0]
						b=opt_choice[1]
						
						#start with latest optimum
						latest_optimum=opt_choice[0][0]+step*2
	return w,b
def predict(feat,w,b):
	#sign(x.w+b)
	
	classification=np.sign(np.dot(feat,w)+b)
	return (classification)
	
# data ={-1:ndarray,1:nparray}
df=pd.read_csv('processed.csv')
ydata=df['class'].values
df_0=df['class']==0
df_0=df[df_0]
df_1=df['class']==1
df_1=df[df_1]
df_0=df_0.drop(['class'],axis=1)
df_1=df_1.drop(['class'], axis=1)

#print(df_0)
#print(df_0.iloc[1])

data={-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}
#w,b=fit(data)
#print("w and b are ",w,"  ",b)
test=np.array([3,8])
print(predict([6,2],[80,-80] ,1))				
		
