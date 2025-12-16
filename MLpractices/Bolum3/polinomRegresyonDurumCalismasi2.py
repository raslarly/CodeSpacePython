from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

house = fetch_california_housing()

x = house.data
y = house.target

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2, random_state=42)

polyF = PolynomialFeatures(degree=2)
xTrainPoly = polyF.fit_transform(xTrain)
xTestPoly = polyF.transform(xTest)

pLinReg = LinearRegression()
pLinReg.fit(xTrainPoly,yTrain)
yPred = pLinReg.predict(xTestPoly)

rmse = root_mean_squared_error(yTest, yPred)

print('Polynomial regression RMSE: {}'.format(rmse))

# in video they found it as 0.681396


LinReg = LinearRegression()
LinReg.fit(xTrain,yTrain)
yPred2 = LinReg.predict(xTest)
rmse2 = root_mean_squared_error(yTest, yPred2)

print('Multi Variable Linear regression RMSE: {}'.format(rmse2))



























































