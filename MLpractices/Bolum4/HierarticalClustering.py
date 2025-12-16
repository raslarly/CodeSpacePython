from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x, _= make_blobs(n_samples=300,centers=3,cluster_std=3,random_state=42)
# plt.figure()
# plt.scatter(x[:,0],x[:,1])
# plt.title('Ornek Veri')

linkageMethods=['ward','single','average','complete']

plt.figure()

for i, linmet in enumerate(linkageMethods,1):
    model = AgglomerativeClustering(n_clusters=4,linkage=linmet)
    clusterLabels = model.fit_predict(x)
    plt.subplot(2,4,i)
    plt.title('{} linkage'.format(linmet.capitalize()))
    dendrogram(linkage(x,method=linmet),no_labels=True)
    plt.xlabel('Veri noktalaraı')
    plt.ylabel('uzaklik')
    plt.subplot(2,4,i+4)
    plt.scatter(x[:,0],x[:,1],c=clusterLabels,cmap='viridis')
    plt.title('{} Linkage clustering'.format(linmet))
    plt.xlabel('X')
    plt.ylabel('y')

















































