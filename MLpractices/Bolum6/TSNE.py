from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', version=1)

x = mnist.data
y = mnist.target.astype(int)

# I reduced the train data size so that my pc can run it
# x1, y1, x2, y2 = train_test_split(x,y, train_size=0.45)

tsne = TSNE(n_components=2)
xTsne = tsne.fit_transform(x)

plt.figure()
plt.scatter(xTsne[:,0],xTsne[:,1],c=y,cmap='tab10',alpha=0.6)
plt.title('TSNE of MNIST dataset')
plt.xlabel('T-SNE dimension 1')
plt.ylabel('T-SNE dimension 2')






















