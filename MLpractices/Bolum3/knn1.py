from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

cancer= load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
df['target'] = cancer.target

# Modelin Train edilmesi

X = cancer.data #featurelar
Y = cancer.target #Target variable

# Train test veri kumesinde ki ayrim
xTrain,xTest,yTrain,yTest=train_test_split(X, Y, test_size=0.3, random_state=42)

# Olceklendirme
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)

# KNN modeli olustur ve train et
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(xTrain,yTrain) # fit fonksiyonu verileri kullanarak KNN algoritmasını egitir

# Sonucların degerlendirilmesi : Test edilmesi
y_pred = KNN.predict(xTest)


accuracy = accuracy_score(yTest,y_pred)

# print(accuracy)
confMX = confusion_matrix(yTest, y_pred)
# print(confMX)

# Hiperparametre ayarlamasi 

"""
    KNN hiperparametreler = k
    k = 1,2,3,..., N
    Accuracy = %A, %B, %C (Dogruluk)
"""

accuracyValues = []
kValues = []
for k in range(1,21):
    knn2 = KNeighborsClassifier(n_neighbors=k)
    knn2.fit(xTrain,yTrain)
    yPred = knn2.predict(xTest)
    dogruluk = accuracy_score(yPred, yTest)
    accuracyValues.append(dogruluk)
    kValues.append(k)

plt.Figure()
plt.plot(kValues,accuracyValues,marker='o',linestyle='-')
plt.grid(True)
plt.title('K degerine gore dogruluk')
plt.xlabel('k degeri')
plt.ylabel('dogruluk')
plt.xticks(kValues)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

sayiList1 = np.sort(5 * np.random.rand(40,1),axis=0) # uniform dagilim featurelar
sayiList2 = np.sin(sayiList1).ravel() # target variables

# Add noise

sayiList2[::5] += 1*(0.5-np.random.rand(8))

# Create test list

tList = np.linspace(0,5,500)[:,np.newaxis]

# plt.plot(sayiList1)
# plt.plot(sayiList1,sayiList2)
# plt.scatter(sayiList1,sayiList2)

for i, weight in enumerate(['uniform','distance']):
    knn3 = KNeighborsRegressor(n_neighbors=5,weights=weight)
    pred2 = knn3.fit(sayiList1,sayiList2).predict(tList)
    plt.subplot(2,1,i+1)
    plt.scatter(sayiList1,sayiList2,color = 'green', label ='data')
    plt.plot(tList,pred2,color='blue',label='tahminler')
    plt.axis('tight')
    plt.legend()
    plt.title(f'KNN regressor weight = {weight}')
    
plt.tight_layout()
plt.show()
    




























