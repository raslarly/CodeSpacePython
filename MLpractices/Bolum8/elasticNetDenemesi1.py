from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error


obese = load_diabetes()

x = obese.data
y = obese.target

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)

eNet = ElasticNet()
eNetParamGrid = {'alpha':[0.1,1,10,100],
                 'l1_ratio':[0.1,0.5,0.3,0.7,0.9]}
eNetGS = GridSearchCV(eNet, eNetParamGrid, cv=5)
eNetGS.fit(xTrain,yTrain)

print('en iyi parametreler: ',eNetGS.best_estimator_,eNetGS.best_score_)

besteNetGS = eNetGS.best_estimator_
yPred = besteNetGS.predict(xTest)
hata = root_mean_squared_error(yTest, yPred)
print(hata)










