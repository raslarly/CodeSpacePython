from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



oli = fetch_olivetti_faces()

"""
2D (64x64) -> 1D (4096) 
"""
plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    # for adding i you see different persons (down there) "[i + {whatever you like }]"
    plt.imshow(oli.images[i+250],cmap='gray') 
    plt.axis('off')
plt.show()

x = oli.data
y = oli.target

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)

rfc = RandomForestClassifier(n_estimators=120,random_state=42) #n_estimators org '100'
rfc.fit(xTrain,yTrain)

yPred = rfc.predict(xTest)

dogruluk = accuracy_score(yTest, yPred)
# print(dogruluk)

# pratik olması açısından n_estimators = 5, 50, 100 ve 500 olacak şekilde deneyecek dogrulul degerlerini
# depolayarak degerlendirecegiz 


for ne in [5,50,100,500]:
    rfc = RandomForestClassifier(n_estimators=ne,random_state=42) #n_estimators org '100'
    rfc.fit(xTrain,yTrain)

    yPred = rfc.predict(xTest)

    dogruluk = accuracy_score(yTest, yPred)
    print(f"ne ={ne} icin dogruluk: ",dogruluk)
    
    

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import numpy as np

chds = fetch_california_housing()

x = chds.data
y = chds.target

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)

rfg = RandomForestRegressor(random_state=42) # Default of number estimator is 100

rfg.fit(xTrain,yTrain)

yPred = rfg.predict(xTest)
mse = mean_squared_error(yTest, yPred)

dogruluk = np.sqrt(mse)
print(dogruluk)





































