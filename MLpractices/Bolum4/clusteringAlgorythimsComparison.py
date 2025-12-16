from sklearn import datasets,cluster
from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

NoS = 1500

noCircle = datasets.make_circles(n_samples=NoS,factor=0.5,noise=0.05)

noMoon = datasets.make_moons(n_samples=NoS,noise=0.05)
noBlobs = datasets.make_blobs(n_samples=NoS)

rastG = np.random.rand(NoS,2), None # Digerleri gibi 2'li donus icin None ekle

clusNames=['MiniBatchKMeans','SpectralClustering','Ward',
           'AgglomativeClustering','DBSCAN','Birch']

color = np.array(['b','g','r','c','m','y'])
dataSets = [noCircle, noMoon, noBlobs, rastG]

plt.figure()

i = 1
for iDS,dataS in enumerate(dataSets):
    
    x,y = dataS
    x = StandardScaler().fit_transform(x)

    twoMean = cluster.MiniBatchKMeans(n_clusters=2)
    wardo = cluster.AgglomerativeClustering(n_clusters=2,linkage='ward')
    spectral = cluster.SpectralClustering(n_clusters=2)
    dbscan = cluster.DBSCAN(eps=0.2)
    avLinkage = cluster.AgglomerativeClustering(n_clusters=2,linkage='average')
    cBirch = cluster.Birch(n_clusters=2)

    clusAlgos =  [twoMean,wardo,spectral,dbscan,avLinkage,cBirch]
    
    for name,algo in zip(clusNames,clusAlgos):
        
        algo.fit(x)
        
        if hasattr(algo,'labels_'):
            yPred = algo.labels_.astype(int)
        else:
            yPred=algo.predict(x)
            
        plt.subplot(len(dataSets),len(clusAlgos),i)

        if iDS==0:
            plt.title(name)
        plt.scatter(x[:,0],x[:,1],color=color[yPred].tolist(),s=10)



        i += 1






















































