from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


opML = fetch_openml('mnist_784',version=1)

x = opML.data
y = opML.target.astype(int)

lda = LinearDiscriminantAnalysis(n_components=2)

xLda = lda.fit_transform(x,y)

plt.figure()
plt.scatter(xLda[:,0],xLda[:,1],c=y,cmap='tab10',alpha=0.6)
plt.title('LDA of MNIST dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar(label='Digits')

#%% LDA vs PCA
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

iris = load_iris()

x = iris.data
y = iris.target

tNames = iris.target_names

pca = PCA(n_components=2)

xPca = pca.fit_transform(x,y)

lda = LinearDiscriminantAnalysis(n_components=2)

xLda = lda.fit_transform(x,y)

colors = ['red','blue','green']

plt.figure()

for color, i, tName in zip(colors,[0,1,2],tNames):
    plt.scatter(xPca[y==i,0],xPca[y==i,1], color=color,alpha=0.8,
                label=tName)
plt.legend()
plt.title('PCA of iris dataset')

plt.figure()
for color, i, tName in zip(colors,[0,1,2],tNames):
    plt.scatter(xLda[y==i,0],xLda[y==i,1], color=color, alpha=0.8,
                label=tName)
plt.legend()
plt.title('LDA of iris dataset')


