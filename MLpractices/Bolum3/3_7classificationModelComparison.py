from sklearn.datasets import make_classification, make_moons,make_circles
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

import numpy as np

x,y=make_classification(n_features=2,n_redundant=0,
                        n_informative=2,n_clusters_per_class=1,
                        random_state=42)
x+=1.2*np.random.uniform(size=x.shape)

# plt.scatter(x[:,0],x[:,1],c=y)

x2,y2=make_moons(noise=0.2,random_state=42)
# plt.scatter(x2[:,0],x2[:,1],c=y2)

x3,y3=make_circles(noise=0.35,factor=0.3,random_state=42)
# plt.scatter(x3[:,0],x3[:,1],c=y3)
datasets =[(x,y),(x2,y2),(x3,y3)]
fig=plt.figure(figsize=(6,9))
i=1
for dsc,ds in enumerate(datasets):
    x4,y4=ds
    if dsc==0:
        colors="darkred"
    elif dsc==1:
        colors="darkblue"
    else:
        colors="darkgreen"
    ax=plt.subplot(len(datasets),1,i)
    ax.scatter(x4[:,0],x4[:,1],c=y4,cmap=plt.cm.coolwarm,
               edgecolors="black")
    i+=1
# plt.show()

names = ["Nearest Neighbors","Linear SVM","Decision Tree",
         "Random Forest","Naive Bayes"]


classifiers = [KNeighborsClassifier(),
               SVC(),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               GaussianNB()  ]

fig = plt.figure(figsize=(15,9))

i=1
for dsc, ds in enumerate(datasets):
    x5,y5=ds
    xTrain,xTest,yTrain,yTest=train_test_split(x5,y5,test_size=0.2,
                                               random_state=42)

    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    cmb=ListedColormap(["darkred","darkblue"])
    
    if dsc==0:
        ax.set_title('Input data')
        
    # plot training data
    ax.scatter(xTrain[:,0],xTrain[:,1],c=yTrain,
               cmap=cmb,edgecolor="black")

    # plot teest data
    ax.scatter(xTrain[:,0],xTrain[:,1],c=yTrain,
               cmap=cmb,edgecolor="black",alpha=0.6)
    i+=1
    
    for name, clf in zip(names,classifiers):
        
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        
        clf = make_pipeline(StandardScaler(),clf)
        clf.fit(xTrain,yTrain)
        score = clf.score(xTest,yTest) # accuracy degerimiz
        DecisionBoundaryDisplay.from_estimator(clf,x5,cmap=plt.cm.RdBu,
                                               alpha=0.7,ax=ax,eps=0.5)
        # plot training data
        ax.scatter(xTrain[:,0],xTrain[:,1],c=yTrain,cmap=cmb,
                   edgecolors="black",alpha=0.5)
        # plot test data
        ax.scatter(xTrain[:,0],xTrain[:,1],c=yTrain,cmap=cmb,
                   edgecolors="black",)
        
        if dsc ==0:
            ax.set_title(name)
        ax.text(x5[:,0].max()+0.35,
                x5[:,0].min()-0.35,
                str(score))
        i+=1

plt.show()
        
    
    


























