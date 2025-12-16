#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 22:49:19 2025

@author: razskler
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


# Veriseti inceleme-analiz
iris = load_iris()

X = iris.data # features
Y = iris.target # target variables

xTrain,xTest,yTrain,yTest = train_test_split(X,Y, test_size=0.2, random_state=42)

# DT modeli olustur ve train et

dtc = DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=42) # criterion ='entropy'
dtc.fit(xTrain,yTrain)

# DT evaluation test asamasi 

yPredict = dtc.predict(xTest)

dogruluk = accuracy_score(yTest, yPredict)
karisiklik = confusion_matrix(yTest, yPredict)

plt.figure(figsize=(15,10))
plotTree = plot_tree(dtc,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
#plt.show()

ozelOnem = dtc.feature_importances_
ozelisim = iris.feature_names
onem = sorted(zip(ozelOnem,ozelisim),reverse=True)

for onemi, isim in onem:
    print(f'{isim}:{onem}')

#print(dogruluk)
#print(karisiklik)

# %%

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()

n_classes = len(iris.target_names)  #3
plot_colors = "ryb"

for pairId, pairs in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    x = iris.data[:,pairs]
    y = iris.target
    
    dtc = DecisionTreeClassifier().fit(x,y)
    
    ax = plt.subplot(2,3,pairId+1)
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
    
    dm = DecisionBoundaryDisplay.from_estimator(dtc,x,
                                                cmap= plt.cm.RdYlBu,
                                                response_method="predict",
                                                ax=ax,
                                                xlabel=iris.feature_names[pairs[0]],
                                                ylabel=iris.feature_names[pairs[1]])
    
    for i, color in zip(range(n_classes),plot_colors):
        idx = np.where(y==i)
        plt.scatter(x[idx,0],x[idx,1],c=color,
                    label = iris.target_names[i],cmap=plt.cm.RdYlBu,
                    edgecolors="black")
        
plt.legend()


# %%

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import numpy as np

diabet = load_diabetes()

x = diabet.data # features
y = diabet.target # tagets

xTrain, xTest, yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)

# karar agaci regression modeli

aReg = DecisionTreeRegressor(random_state=42)
aReg.fit(xTrain,yTrain)

yPred = aReg.predict(xTest)

kokd = mean_squared_error(yTest, yPred)
print(dogruluk)

kka = np.sqrt(kokd)
print(kka)

# %%

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

import numpy as np

x = np.sort(5*np.random.rand(80,1),axis=0)
y = np.sin(x).ravel()
y[::5] += 0.5*(0.5-np.random.rand(16))

# plt.scatter(x,y)

rg1 = DecisionTreeRegressor(max_depth=2)
rg2 = DecisionTreeRegressor(max_depth=5)
rg3 = DecisionTreeRegressor(max_depth=15)

rg1.fit(x,y)
rg2.fit(x,y)
rg3.fit(x,y)

xTest = np.arange(0,5,0.05)[:,np.newaxis]
yPred1 = rg1.predict(xTest)
yPred2 = rg2.predict(xTest)
yPred3 = rg3.predict(xTest)

plt.figure()
plt.plot(x,y,c='red',label='data')
plt.scatter(x,y,c='red',label='data')
# use lw as a shortcut to linewidth
plt.plot(xTest,yPred1,color='blue',label='derinlik 2',lw=2) 
plt.plot(xTest,yPred2,color='green',label='derinlik 5',lw=2)
plt.plot(xTest,yPred3,color='yellow',label='derinlik d1',lw=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()

























