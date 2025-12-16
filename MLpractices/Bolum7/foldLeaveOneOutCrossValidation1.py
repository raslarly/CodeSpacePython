from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, KFold, LeaveOneOut,
                                     GridSearchCV)
from sklearn.tree import DecisionTreeClassifier



iris = load_iris()

x=iris.data
y=iris.target

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,
                                             random_state=42)

dtc = DecisionTreeClassifier()
dtcParamList = {'max_depth':[3,5,7]}

# kFold Grid Search
kF = KFold(n_splits=10)
dtcGS = GridSearchCV(dtc, dtcParamList, cv=kF)
dtcGS.fit(xTrain,yTrain)
enIyiSonuckF = dtcGS.best_estimator_, dtcGS.best_score_
print(f'KF En iyi sonuc: {enIyiSonuckF}')

# Leave One Out 
loo = LeaveOneOut()
dtclooGS = GridSearchCV(dtc, dtcParamList, cv=loo)
dtclooGS.fit(xTrain,yTrain)
enIyiSonucloo = dtclooGS.best_estimator_, dtclooGS.best_score_
print(f'LOO En iyi sonuc: {enIyiSonucloo}')















