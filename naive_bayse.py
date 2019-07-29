#naive bayse
import json
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
def naive_bayse(xtrain):
    #p= p(x|y)/p(x)
    pass
    

#training(xtrain)
def train(xtrain,ytrain):
	
    df=pd.concat([ytrain,xtrain],axis=1)
    #print(df.head(5))
    

    
    out=defaultdict()
    total=xtrain.shape[0]
    #print(np.unique(ytrain))
    rdf=pd.DataFrame()
    for clas in np.unique(ytrain):
        df_0=df['class']==clas
        df_0=(df[df_0]).drop(['class'],axis=1)
        total0=df_0.shape[0]
        Tp=pd.DataFrame()
        Tdf_=pd.DataFrame()
        for col in df_0.columns:
                
            df_=df_0[col].value_counts()
            #print(df_.head(5))
            p=np.array([df_[i]/total0 for i in df_.index ])#,columns="P(in class) "+str(col)
            
            p=pd.DataFrame(p,columns=["P("+str(col)+") "])
            df_=pd.DataFrame(df_,columns=[str(col)])
            #print(df_.head(5))
            #print(p.shape,df_.shape)
            Tp=pd.concat([Tp,p],axis=1)
            Tdf_=pd.concat([Tdf_,df_],axis=1)
            
            #print(Tdf_.head(6))
        #c=pd.DataFrame([clas]*Tp.shape[0],columns=['class'])	
        rdf_=pd.concat([Tdf_,Tp],axis=1)
        rdf_=rdf_.fillna(0)
        out.update({str(clas):rdf_.to_dict()})

      
    with open("training.json",'w')  as f:
        json.dump(out,f, indent=4)
    return True
def execute(x_pred):
    with open("training.json",'r')  as f:
        data=json.load(f)
    P=1
    for key in data.keys():
        i=0
        for kk in data[key]:
            total=sum( [len(data[k]) for k in data.keys()])
            pC= len(data[key])/total
            
            pV_C=float(data[key]["P("+kk+") "][str(x_pred[i])])
            
            pV=float_round(sum( [int(data[k][kk][str(x_pred[i]) ]) for k in data.keys()]) /total,2)
            if (pV_C !=0 and (pC !=0 and pV!=0)):
                P*=(pV_C*pC/pV)
            i+=1
            if i>=len(data[key])*0.5:
                break
        print("prob ",P)
df=pd.read_csv('processed.csv')
ydata=df['class']
xdata=df.drop(['class'],axis=1)

#print(df_0)
#print(df_1)
#train(xdata,ydata)
print(xdata.head(5))
execute(list(xdata.iloc[1]))		
	
	
