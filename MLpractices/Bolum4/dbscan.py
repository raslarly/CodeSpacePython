from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

x,_=make_circles(n_samples=1000,factor=0.5,noise=0.08,random_state=42)
# plt.figure()
# plt.scatter(x[:,0],x[:,1])

dbscan = DBSCAN(eps=0.15,min_samples=15)
clusterLabels=dbscan.fit_predict(x)
plt.figure()
plt.scatter(x[:,0],x[:,1],c=clusterLabels,cmap='viridis')
plt.title('DBSCAN sonuclari')

















































