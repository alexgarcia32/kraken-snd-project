import os
import numpy as np
import pandas as pd
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

#from scipy.spatial import distance
#import scipy
#import copy

#from sklearn import metrics
from sklearn.cluster import DBSCAN


mean = [0, 1]
cov = [[1, 0.5], [1, 2]]

mean1 = [5, -6]
cov1 = [[0.5, 0.25], [0.5, 1]]


random.seed(6)
x, y = np.random.multivariate_normal(mean, cov, 1000).T
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T



plt.plot(x, y, 'x')
plt.plot(x1, y1, 'x')
#plt.axis('equal')
plt.show()


dat = {'x':[x],'y':[y]}
data = pd.DataFrame(np.random.multivariate_normal(mean, cov, 500))
dat1 = {'x':x1,'y':y1}
data1 = pd.DataFrame(np.random.multivariate_normal(mean1, cov1, 500))


epsilon = 0.8
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

graph_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2', 'PC3'])
graph_df['News'] = graph_df.index
graph_df.set_index("News", inplace=True)
graph_df = graph_df.append(pca_fake_df)

# We split labels in two: labels of clustered data and labels of noise points
labels_clustered = dbscan_labels[dbscan_labels >= 0]
# labels_noise = dbscan_labels[dbscan_labels < 0]



plt.scatter(data.iloc[dbscan_labels >= 0, 0], data.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50, cmap='viridis')
plt.scatter(data.iloc[dbscan_labels < 0, 0], data.iloc[dbscan_labels < 0, 1], c='r',marker='*', s=50)
plt.title("DBSCAN: Visualization of the clustered data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()




### PRIMER GRÁFICO
mean = [0, 1]
cov = [[1, 0.25], [0.25, 1]]
data = pd.DataFrame(np.random.multivariate_normal(mean, cov, 500))

epsilon = 0.8
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

labels_clustered = dbscan_labels[dbscan_labels >= 0]

plt.scatter(data.iloc[dbscan_labels >= 0, 0], data.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50, cmap='viridis')
plt.scatter(data.iloc[dbscan_labels < 0, 0], data.iloc[dbscan_labels < 0, 1], c='r',marker='*', s=50)
plt.xlim([-7, 7])
plt.ylim([-7, 8])
plt.show()


### SEGUNDO GRÁFICO
mean = [0, 1]
cov = [[1, 0.25], [0.25, 1]]
data = pd.DataFrame(np.random.multivariate_normal(mean, cov, 500))
mean1 = [5, -6]
cov1 = [[1, 0.25], [0.25, 1]]
data1 = pd.DataFrame(np.random.multivariate_normal(mean1, cov1, 500))

data2 = data.append(data1)

epsilon = 0.8
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data2)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

labels_clustered = dbscan_labels[dbscan_labels >= 0]

plt.scatter(data2.iloc[dbscan_labels >= 0, 0], data2.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50, cmap='viridis')
plt.scatter(data2.iloc[dbscan_labels < 0, 0], data2.iloc[dbscan_labels < 0, 1], c='r',marker='*', s=50)
plt.xlim([-5, 10])
plt.ylim([-11, 6])
plt.show()





### TERCER GRÁFICO
random.seed(6)
mean = [0, 1]
cov = [[1, 0.25], [0.25, 1]]
data = pd.DataFrame(np.random.multivariate_normal(mean, cov, 500))
mean1 = [5, -6]
cov1 = [[1, 0.25], [0.25, 1]]
data1 = pd.DataFrame(np.random.multivariate_normal(mean1, cov1, 500))


mean_out1 = [-1, -7]
cov_out1 = [[1, 0.25], [0.25, 1]]
data_out1 = pd.DataFrame(np.random.multivariate_normal(mean_out1, cov_out1, 1))
mean_out2 = [7, 3]
cov_out2 = [[1, 0.25], [0.25, 1]]
data_out2 = pd.DataFrame(np.random.multivariate_normal(mean_out2, cov_out2, 2))


data_out1a = data.append(data1)
data_out2a = data_out1a.append(data_out1)
data_out = data_out2a.append(data_out2)

epsilon = 0.8
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data_out)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

labels_clustered = dbscan_labels[dbscan_labels >= 0]

plt.scatter(data_out.iloc[dbscan_labels >= 0, 0], data_out.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50, cmap='viridis')
plt.scatter(data_out.iloc[dbscan_labels < 0, 0], data_out.iloc[dbscan_labels < 0, 1], c='r',marker='*', s=50)
plt.xlim([-5, 10])
plt.ylim([-11, 6])
plt.show()




