import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# y = a0 + a1*x
# y = a0 +a1*x1 + a2*x2 + ... + an*xn -> multi variable lineer regression
# y = a0 + a1*x1 + a2*x2

xu = np.random.rand(100,2)

coef = np.array([3,5])

# y = 0 + np.dot(xu,coef)

y =  12 +5*np.random.rand(100) + np.dot(xu,coef)

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(xu[:,0],xu[:,1],y)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')

linReg = LinearRegression()

linReg.fit(xu,y)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xu[:,0],xu[:,1],y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')


xu1,xu2 = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))

yPred = linReg.predict(np.array([xu1.flatten(),xu2.flatten()]).T)
ax.plot_surface(xu1,xu2,yPred.reshape(xu1.shape),alpha=0.3)
plt.title('Multi linear regression')

print('katsayilar: ', linReg.coef_)
print('Kesisim: ', linReg.intercept_)

# %%

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

obese = load_diabetes()

x = obese.data
y = obese.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)

linReg = LinearRegression()

linReg.fit(xTrain,yTrain)

yPred = linReg.predict(xTest)

rmse = root_mean_squared_error(yTest,yPred)

print('dogruluk orani: ',rmse)





































