from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()

x = iris.data
y = iris.target

pca = PCA(n_components=2) # 2 temel bilesen (PC)

xPca = pca.fit_transform(x)


plt.figure()

for i in range(len(iris.target_names)):
    plt.scatter(xPca[y==i, 0], xPca[y==i, 1], label=iris.target_names[i])

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.legend()

# %%

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


iris = load_iris()

x = iris.data
y = iris.target

pca = PCA(n_components=3) # 3 adet temel bilesen
xPca = pca.fit_transform(x)

fig = plt.figure(1, figsize=(8,6))

ax = fig.add_subplot(111, projection='3d',elev=-150,azim=110)

ax.scatter(xPca[:,0],xPca[:,1],xPca[:,2],c=y, s=40)

ax.set_title('first three PCA dimensions of iris dataset')
ax.xlabel('1st Eigenvector')
ax.ylabel('2nd Eigenvector')
ax.zlabel('3rd Eigenvector')



































