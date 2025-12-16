from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # SupportVectorClassifier/Regressor
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

digits = load_digits()

dig,axes= plt.subplots(nrows=2,ncols=5,figsize=(10,5),
                       subplot_kw={"xticks":[],"yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap="binary",interpolation="nearest")
    ax.set_title(digits.target[i])
#plt.show()

x =digits.data
y=digits.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)

svmc = SVC(kernel="linear",random_state=42)
svmc.fit(xTrain,yTrain)
yPred=svmc.predict(xTest)

rapor = classification_report(yTest,yPred)
print(rapor)



