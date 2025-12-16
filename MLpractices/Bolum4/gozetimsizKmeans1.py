from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


x, _ =make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)


plt.figure()
plt.scatter(x[:,0],x[:,1])
plt.title('Ornek veri')

kMeans = KMeans(n_clusters=4)
kMeans.fit(x)

labels = kMeans.labels_

plt.figure()

# the scatter fuction colors the clusters using the 'c=labes'
plt.scatter(x[:,0],x[:,1],c=labels,cmap='viridis')

center = kMeans.cluster_centers_
plt.scatter(center[:,0],center[:,1],c='red',marker='X')
plt.title('K-Means')















































