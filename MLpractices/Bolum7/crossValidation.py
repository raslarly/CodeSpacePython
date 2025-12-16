from sklearn.model_selection import (train_test_split, GridSearchCV, 
RandomizedSearchCV)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
x = iris.data
y = iris.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,
                                           random_state=42)

dt = DecisionTreeClassifier()

dtParamGrid = {'max_depth':[3,5,7],
               'max_leaf_nodes':[None,5,10,20,30,50]}

nCV = 3
dtGS = GridSearchCV(dt, dtParamGrid,cv=nCV)
dtGS.fit(xTrain,yTrain)
print('KNN Grid Search Best parameter: ', dtGS.best_params_)
print('KNN Grid Search Best Score: ', dtGS.best_score_)

dtRS = RandomizedSearchCV(dt, dtParamGrid, n_iter=10)
dtRS.fit(xTrain,yTrain)
print('KNN Random Search Best parameter: ', dtRS.best_params_)
print('KNN Random Search Best Score: ', dtRS.best_score_)

for meanScore, params in zip(dtGS.cv_results_['mean_test_score'],
                             dtGS.cv_results_['params']):
    print(f'ortalama test skoru: {meanScore}, parametreler: {params}')

cvSonuc = dtGS.cv_results_

for i, params in enumerate((cvSonuc['params'])):
    print(f'parametreler: {params}')
    
    for j in range(nCV):
        dogruluk = cvSonuc[f'split{j}_test_score'][i]
        print(f'\tFold: {j+1} - Accuracy: {dogruluk}')
















