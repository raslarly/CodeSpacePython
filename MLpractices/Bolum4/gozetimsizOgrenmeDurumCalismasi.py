import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.cluster import KMeans
#from sklearn.model_selection import train_test_split                
from scipy.cluster.hierarchy import dendrogram, linkage


# Verisetini bulamadim her nedense :) ben de kendim elde ettim sonuclar illa ki
# farkli olacak

# df1 = pd.read_csv('customer_data.csv')

# df1['spendingScore'] = df1['purchase_frequency'] * df1['spending']

# df2 = df1[['income','spendingScore']]

# df2.to_csv('MusteriVerisi.csv',index=False)

# Assagida egitimde pickel formati kullanildigindan fromatta degisiklige 
# gidiyorum 

# df = pd.read_csv('MusteriVerisi.csv')
# df.to_pickle('Musteriler.pkl')

data = pd.read_pickle('Musteriler.pkl')

x = data.values

# This was just a practise i dont need to do this

# x[:,0] = np.abs(2*min(x[:,0])) +x[:,0]
# x[:,1] = np.abs(2*min(x[:,1])) +x[:,1]


# plt.figure()
# plt.scatter(x[:,0],x[:,1], s=50, alpha=0.7,edgecolors='k')
# plt.title('Musteri Segmentasyonu')
# plt.xlabel('income')
# plt.ylabel('spoending Score')

# print(max(x[:,0]))

kms1 = KMeans(n_clusters=5, )
kms1.fit(x)

clusLabs = kms1.labels_
centers = kms1.cluster_centers_

plt.figure(figsize=(15,6))
# 1 satir ve iki sutundan olusan bir subplot olsuturalim
plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c=clusLabs ,s=50,alpha=0.7,edgecolors='k')
plt.scatter(centers[:,0],centers[:,1],c=range(len(centers)),s=200,alpha=0.8,
            marker='X',cmap='viridis')
plt.title('KMeans Musteri Segmentasyonu')
plt.xlabel('income')
plt.ylabel('spending Score')

linMatx = linkage(x,method='ward')

plt.subplot(1,2,2)
dendrogram(linMatx)
plt.title('Dendrogram Musteri segmentasyonu')
plt.xlabel('Veri noktalari')
plt.ylabel('Uzaklik')



















































































