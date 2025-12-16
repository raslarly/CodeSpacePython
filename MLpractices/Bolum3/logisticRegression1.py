# https://archive.ics.uci.edu/dataset/45/heart+disease

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

import warnings


warnings.filterwarnings("ignore")
kalph = fetch_ucirepo(name='heart disease')

df = pd.DataFrame(data = kalph.data.features)

df["target"] = kalph.data.targets
# bos verileri cikariyoruz veriyi temizliyoruz
if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan")

x = df.drop(["target"],axis=1).values #values x'i bir np array'a cevirmek icin
y = df.target.values

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.1,random_state=42)

lReg = LogisticRegression(penalty="l2",C=1,solver="lbfgs",max_iter=100)
lReg.fit(xTrain,yTrain)

dogruluk = lReg.score(xTest,yTest)
print(dogruluk)








