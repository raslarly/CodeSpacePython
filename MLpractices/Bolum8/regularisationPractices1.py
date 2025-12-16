from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import root_mean_squared_error


obese = load_diabetes()

x = obese.data
y = obese.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)

# Ridge

rid = Ridge()
ridParamGrid = {'alpha':[0.1,1,10,100]}
ridGS = GridSearchCV(rid, ridParamGrid, cv=5)
ridGS.fit(xTrain,yTrain)
print('Ridge en iyi parametreler: ', ridGS.best_params_, ridGS.best_score_)
bestRidGS = ridGS.best_estimator_
yPred1 = bestRidGS.predict(xTest)

hata1 = root_mean_squared_error(yTest, yPred1)
print(hata1)


las = Lasso()
lasParamGrid = {'alpha':[0.1,1,10,100]}
lasGS = GridSearchCV(las, lasParamGrid)
lasGS.fit(xTrain,yTrain)
print('Lassso en iyi parametreler: ',lasGS.best_params_, lasGS.best_score_)
bestLasGS = lasGS.best_estimator_
yPred2 = bestLasGS.predict(xTest)
hata2 = root_mean_squared_error(yTest, yPred2)
print(hata2)































