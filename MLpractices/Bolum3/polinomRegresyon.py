import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = 4 * np.random.rand(100,1)

# 2 + 3*x^2
y = 2 + 3*x**2 + (5.5 * np.random.rand(100,1))

#plt.plot(x,y)
#plt.scatter(x,y)


poly = PolynomialFeatures(degree=2)
xPoly = poly.fit_transform(x)

polyReg = LinearRegression()
polyReg.fit(xPoly,y)

plt.scatter(x,y,color='blue')

xTest = np.linspace(0,4,100).reshape(-1,1)

xTestPoly = poly.transform(xTest)

yPred = polyReg.predict(xTestPoly)


plt.plot(xTest,yPred,color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polinom regresyon modeli')























































































