from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, GridSearchCV, 
RandomizedSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


import numpy as np



iris = load_iris()

x = iris.data
y = iris.target

xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.2,random_state=42)

# KNN Classifier
knn = KNeighborsClassifier()
knnParamGrid = {'n_neighbors':np.arange(2,31)}

knnGS = GridSearchCV(knn, knnParamGrid)
knnGS.fit(xTrain,yTrain)
print('KNN Grid Search Best parameter: ', knnGS.best_params_)
print('KNN Grid Search Best Score: ', knnGS.best_score_)

knnRS = RandomizedSearchCV(knn, knnParamGrid, n_iter=10)
knnRS.fit(xTrain,yTrain)
print('KNN Random Search Best parameter: ', knnRS.best_params_)
print('KNN Random Search Best Score: ', knnRS.best_score_)

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dtParamGrid = {'max_depth':np.arange(2,10),
               'max_leaf_nodes':[None,5,10,20,30,50]}

dtGS = GridSearchCV(dt, dtParamGrid)
dtGS.fit(xTrain,yTrain)
print('KNN Grid Search Best parameter: ', dtGS.best_params_)
print('KNN Grid Search Best Score: ', dtGS.best_score_)

dtRS = RandomizedSearchCV(dt, dtParamGrid, n_iter=10)
dtRS.fit(xTrain,yTrain)
print('KNN Random Search Best parameter: ', dtRS.best_params_)
print('KNN Random Search Best Score: ', dtRS.best_score_)

# Support Vector Machine

svc = SVC()

svcParamGrid = {'C':[0.1,1,10,100],
                'gamma':[0.1,0.01,0.001,0.0001]}

svcGS = GridSearchCV(svc, svcParamGrid)
svcGS.fit(xTrain,yTrain)
print('SVC Grid Search Best parameter: ', svcGS.best_params_)
print('SVC Grid Search Best Score: ', svcGS.best_score_)

svcRS = RandomizedSearchCV(svc, svcParamGrid, n_iter=10)
svcRS.fit(xTrain,yTrain)
print('SVC Random Search Best parameter: ', svcGS.best_params_)
print('SVC Random Search Best Score: ', svcGS.best_score_)

















