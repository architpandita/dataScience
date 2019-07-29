import numpy as np
import pandas as pd
import json
data=pd.read_csv("mushroom-classification/mushrooms.csv")
#print(data)
#print(data.info())
data=data.drop(['gill-attachment','veil-type','veil-color'],axis=1)
mapper={}
for col in data.columns:
	uni=data[col].unique()
	mapper[col]={ uni[i]:i for i in range(len(uni)) }
	#print(data[col].value_counts())
	
#print(mapper)

with open('mapper.json','w') as f:
	json.dump(mapper,f, indent=4)

for col in data.columns:
	data[col]=data[col].map(mapper[col])

data.to_csv('processed.csv',index=False)




