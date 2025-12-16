from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# veri olustur

v1 = np.random.rand(100,1)

# v2= 3 + 4v1

v2 = 3 + 4*v1
v2 = v2+ np.random.rand(100,1)*5

linReg=LinearRegression()

linReg.fit(v1,v2)

plt.figure()
plt.scatter(v1,v2)
plt.plot(v1,linReg.predict(v1),color='red',alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lineer Regresyon')

# y = 3 + 4x -> y = a0 + a1x

a0 = linReg.intercept_

# y = 3 + 4x -> y = a0 + a1x

a1 = linReg.coef_[0][0] # X ekseninin katsayisi (egim)
print(f'a0:{a0}a1:{a1}')

for i in range(100):
    vp1 = a0 + a1
    plt.plot(v1,vp1,color='green',alpha=0.7)

# %%

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt



obeseX,obeseY=load_diabetes(return_X_y=True)

obeseX = obeseX[:,np.newaxis,2]

obeseXTrain = obeseX[:-20]
obeseXTest = obeseX[-20:]

obeseYTrain = obeseY[:-20]
obeseYTest = obeseY[-20:]

linReg = LinearRegression()

linReg.fit(obeseXTrain,obeseYTrain)

obeseYPred = linReg.predict(obeseXTest)

mse = mean_squared_error(obeseYTest,obeseYPred)
r2 = r2_score(obeseYTest,obeseYPred)

print(mse)
print(r2)
plt.scatter(obeseXTest,obeseYTest,color='black')
plt.plot(obeseXTest,obeseYPred,color='blue')


























