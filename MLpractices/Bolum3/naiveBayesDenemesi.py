from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

iris = load_iris()

x=iris.data
y=iris.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)
nbc = GaussianNB()
nbc.fit(xTrain,yTrain)
yPred = nbc.predict(xTest)
dogruluk =classification_report(yTest, yPred)
print(dogruluk)
