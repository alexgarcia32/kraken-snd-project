import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial import distance
import scipy
import copy

from sklearn.cluster import DBSCAN
from sklearn import metrics

from yellowbrick.cluster import KElbowVisualizer

# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

# Fake matrix
# aa = np.random.random((5, 5))
# sim_m = np.matrix(aa)




################################################################################################
###########                       FUNCTIONS SILHOUETTE + KMEANS                      ###########
################################################################################################

def NumClustersSelection(data, range_clusters):
    # Empty list for silhouette values
    silhouette_score_values = list()

    # To select optimal number of clusters
    for i in range_clusters:
        clusterer = KMeans(n_clusters=i, random_state=10)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_score_values.append(
            silhouette_score(data, cluster_labels, metric='euclidean', sample_size=None, random_state=None))

    n_cluster = range_clusters[silhouette_score_values.index(max(silhouette_score_values))]
    silhouette_max_value = max(silhouette_score_values)

    return n_cluster, silhouette_max_value, silhouette_score_values



def ClusterResults(data, n_cluster):
    # To obtain cluster results using the optimal number of clusters
    clusterer = KMeans(n_clusters=n_cluster, random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    centers = clusterer.cluster_centers_

    # Silhouette values
    silhouette_values = silhouette_samples(data, cluster_labels)
    # Global mean
    silhouette_avg = silhouette_score(data, cluster_labels)

    # Labels and news
    lab = pd.DataFrame(cluster_labels, columns=["Labels"])
    lab['News'] = data.index
    lab.set_index("News", inplace=True)
    return cluster_labels, centers, silhouette_values, silhouette_avg, lab


def outlier_distance_detection(data, silhouette_min, size_min, silhouette_values, cluster_labels, n_cluster, centers):
    X1 = copy.deepcopy(data)

    ## Array for outlier detection
    outlier_detection = pd.DataFrame(np.zeros(shape=(len(data), 1)))
    outlier_detection['News'] = data.index
    outlier_detection.set_index("News", inplace=True)
    # outlier_detection_df=pd.DataFrame(outlier_detection)

    ## We first identify outliers.
    # We consider a point (news) is an outlier if one of the following conditions applies:
    # 1. Their silhouette value is < silhouette_min
    # 2. They belong to a cluster whose size is < size_min
    # 3. High contribution to the variance of their cluster

    # 1. Their silhouette value is < silhouette_min
    index_silhouette_min = np.where(silhouette_values < silhouette_min)
    X1.iloc[index_silhouette_min] = 100

    # 2. They belong to a cluster whose size is < size_min
    size_clusters = np.array(np.unique(cluster_labels, return_counts=True)).T
    small_cluster = size_clusters[size_clusters[:, 1] <= size_min, 0]
    a = list(cluster_labels)
    index_small_cluster = [i for i in range(len(a)) if a[i] in small_cluster]
    X1.iloc[index_small_cluster, :] = 100

    # 3. High contribution to the variance of their cluster
    index_contrib = []
    for i in range(n_cluster):
        cluster_i = data.iloc[(cluster_labels == i), :]
        dist_all = scipy.spatial.distance.cdist(cluster_i, centers[i].reshape(1, len(data)), 'euclidean')
        dist_sum = dist_all.sum()
        if dist_sum != 0:
            # Contribution of each point (news)
            individual_contrib = dist_all / dist_sum
            # Q1, Q3, IQR
            q1 = np.percentile(individual_contrib, 25)
            q3 = np.percentile(individual_contrib, 75)
            IQR = q3 - q1
            # Outlier: >q3+1.5*IQR (only high contribution)
            index_contrib.append(cluster_i[individual_contrib > (q3 + 1.5 * IQR)].index)
            X1.loc[index_contrib[i], :] = 100

    ## We calculate distance between outlier and the closest not outlier point
    # Size
    dist_small = scipy.spatial.distance.cdist(data.iloc[index_small_cluster, :], X1, 'euclidean')
    outlier_detection.iloc[index_small_cluster] = dist_small.min(axis=1).reshape(len(dist_small.min(axis=1)), 1)
    # Silhouette value
    dist_silhouette = scipy.spatial.distance.cdist(data.iloc[index_silhouette_min], X1, 'euclidean')
    outlier_detection.iloc[index_silhouette_min] = dist_silhouette.min(axis=1).reshape(len(dist_silhouette.min(axis=1)), 1)
    # Contribution
    for i in range(n_cluster):
        dist_contrib = scipy.spatial.distance.cdist(data.loc[index_contrib[i], :], X1, 'euclidean')
        outlier_detection.loc[index_contrib[i]] = dist_contrib.min(axis=1).reshape(len(dist_contrib.min(axis=1)), 1)

    return outlier_detection

def graph_kmeans_silhouette(data,labs,n_cluster,cluster_labels,silhouette_values,silhouette_avg):

    # We save labels
    labels = pd.DataFrame()
    labels = pd.concat([labels, labs], axis=1, sort=False)

    ### PCA for graphic axis
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    principalDf['News'] = labs.index
    principalDf.set_index("News", inplace=True)
    # PCA + labels
    finalDf = pd.concat([principalDf, labs["Labels"]], axis=1, sort=False)

    # Silhouette graph
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(data) + (n_cluster + 1) * 10])
    y_lower = 10
    for i in range(n_cluster):
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(finalDf["Labels"].astype(float) / n_cluster)
    ax2.scatter(finalDf.iloc[:, 0], finalDf.iloc[:, 1], marker='.', s=150, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_cluster),
                    fontsize=14, fontweight='bold')

    plt.show()


    # Outliers graph
    news = list(finalDf.index)
    #fig, ax = plt.subplots()
    colors = cm.nipy_spectral(finalDf["Labels"].astype(float) / n_cluster)
    plt.scatter(finalDf.iloc[:, 0], finalDf.iloc[:, 1], marker='.', s=150, lw=0, alpha=0.8,
                c=colors, edgecolor='k')

    for i, txt in enumerate(news):
        plt.annotate(txt, (finalDf.iloc[i, 0], finalDf.iloc[i, 1]))
    plt.title("Kmeans: Clustered data with labels")
    plt.show()

    return var_exp



################################################################################################
###########                             FUNCTIONS DBSCAN                             ###########
################################################################################################




import pandas as pd
from sklearn.datasets import load_iris
import copy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def dbscan_clustering(data,epsilon,min_samples,graph_labels):
    # DBSCAN algorithm requires 2 parameters (DBSCAN doesn't have centers):
    # epsilon , which specifies how close points should be to each other to be considered a part of a cluster;
    # and min_samples , which specifies how many neighbors a point should have to be included into a cluster.

    # data is a dataframe
    # if graph_labels="all", graph will have labels for all points, otherwise only for noise points

    # We compute DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(data)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

    # Graph
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_

    graph_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2'])
    graph_df['News'] = data.index
    graph_df.set_index("News", inplace=True)
    # We split labels in two: labels of clustered data and labels of noise points
    labels_clustered = dbscan_labels[dbscan_labels >= 0]
    #labels_noise = dbscan_labels[dbscan_labels < 0]
    # For labels
    news = list(graph_df.index)

    if graph_labels=="all":# label for all points
        plt.scatter(graph_df.iloc[dbscan_labels >= 0, 0], graph_df.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50,
                    cmap='viridis')
        plt.scatter(graph_df.iloc[dbscan_labels < 0, 0], graph_df.iloc[dbscan_labels < 0, 1], c='k', marker='*', s=50)
        for i, txt in enumerate(news):
            plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
        plt.title("DBSCAN: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()
    else: # label for noise points
        plt.scatter(graph_df.iloc[dbscan_labels >= 0, 0], graph_df.iloc[dbscan_labels >= 0, 1], c=labels_clustered, s=50, cmap='viridis')
        plt.scatter(graph_df.iloc[dbscan_labels < 0, 0], graph_df.iloc[dbscan_labels < 0, 1], c='k', marker='*', s=50)
        for i, txt in enumerate(news):
            if dbscan_labels[i] == -1:
                plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
        plt.title("DBSCAN: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

    return dbscan_labels,dbscan_n_clusters,dbscan_n_noise,var_exp


################################################################################################
###########                       FUNCTIONS SPECTRAL CLUSTERING                      ###########
################################################################################################
#https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
#data = iris

from sklearn.cluster import SpectralClustering

def spectral_clustering(data,n_cluster,n_neigh,graph_labels):
    # assign_labels=‘discretize’ it is less sensitive to random initialization than kmeans
    # Instead of using knn, we can use kernels
    model = SpectralClustering(n_clusters=n_cluster,n_init=10, affinity='nearest_neighbors',
                               n_neighbors=n_neigh,assign_labels='kmeans') # assign_labels=‘discretize’
    spectral_labels = model.fit_predict(data)

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_

    graph_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2'])
    graph_df['News'] = data.index
    graph_df.set_index("News", inplace=True)
    # For labels
    news = list(graph_df.index)

    if graph_labels == "all":
        plt.scatter(graph_df.iloc[:, 0], graph_df.iloc[:, 1], c=spectral_labels, s=50, cmap='viridis')
        for i, txt in enumerate(news):
            plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
        plt.title("Spectral clustering: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()
    else:
        plt.scatter(graph_df.iloc[:, 0], graph_df.iloc[:, 1], c=spectral_labels, s=50, cmap='viridis')
        plt.title("Spectral clustering: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

    return spectral_labels, var_exp

################################################################################################
###########                           FUNCTIONS MEAN SHIFT                           ###########
################################################################################################
# https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
# Only one parameter: the kernel bandwidth parameter

from sklearn.cluster import MeanShift
# MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None)
# if bandwidth=None, it's estimated using sklearn.cluster.estimate_bandwidth
# seeds: Seeds used to initialize kernels. If not set, the seeds are calculated by clustering.get_bin_seeds



def MeanShifClustering(data,bandwidth_value,minBinFreq,graph_labels):
    # MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None)
    # if bandwidth=None, it's estimated using sklearn.cluster.estimate_bandwidth
    # seeds: Seeds used to initialize kernels. If not set, the s
    ms = MeanShift(bandwidth=bandwidth_value, bin_seeding=True, min_bin_freq=minBinFreq, cluster_all=False).fit(data)
    ms_labels = ms.labels_
    ms_n_clusters = len(set(ms_labels)) - (1 if -1 in ms_labels else 0)
    ms_n_noise = list(ms_labels).count(-1)

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_

    graph_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2'])
    graph_df['News'] = data.index
    graph_df.set_index("News", inplace=True)
    # For labels
    news = list(graph_df.index)

    if graph_labels == "all":
        plt.scatter(graph_df.iloc[:, 0], graph_df.iloc[:, 1], c=ms_labels, s=50, cmap='viridis')
        for i, txt in enumerate(news):
            plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
        plt.title("Mean Shift clustering: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()
    else:
        plt.scatter(graph_df.iloc[:, 0], graph_df.iloc[:, 1], c=ms_labels, s=50, cmap='viridis')
        plt.title("Mean Shift clustering: Visualization of the clustered data")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

    return ms_labels, ms_n_clusters, ms_n_noise, var_exp



################################################################################################
###########                    FUNCTIONS HIERARCHICAL CLUSTERING                     ###########
################################################################################################
# from sklearn.cluster import AgglomerativeClustering

# cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
# cluster.fit_predict(data)
#cluster.labels_

#plt.figure(figsize=(10, 7))
#plt.scatter(data.iloc[:,0], data.iloc[:,1], c=cluster.labels_, cmap='rainbow')
#plt.show()

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(sim_m, method='ward'))
plt.show()

from scipy.cluster.hierarchy import fcluster
fl = fcluster(shc.linkage(sim_m, method='ward'), 3, criterion='maxclust')

################################################################################################
###########                        FUNCTIONS ISOLATION FOREST                        ###########
################################################################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

## TERMINAR




################################################################################################
################################################################################################
###########################                   RUNNING                ###########################
################################################################################################
################################################################################################


path = 'C:/Users/Carmen/PycharmProjects/classifier/data'
os.chdir(path)
sim_m = pd.read_csv("MatrizSimilaridad_Dia_2019-09-26.csv",sep=";",decimal=",",header=0,index_col=0)
sim_m.index.name = 'News'
sim_m_basic = pd.read_csv("SIMPLE_MatrizSimilaridad_Dia_2019-09-26.csv",sep=";",decimal=",",header=0,index_col=0)
sim_m_basic.index.name = 'News'



## Silhouette + Kmeans
#data = sim_m
range_clusters = range(2,10)
n_cluster, silhouette_max_value, silhouette_score_values = NumClustersSelection(sim_m, range_clusters)
cluster_labels, centers, silhouette_values, silhouette_avg, labs = ClusterResults(sim_m, n_cluster)
# Outlier
silhouette_min = 0
size_min = 5
out_dist = outlier_distance_detection(sim_m, silhouette_min, size_min, silhouette_values, cluster_labels, n_cluster, centers)

var_exp = graph_kmeans_silhouette(sim_m,labs,n_cluster,cluster_labels,silhouette_values,silhouette_avg)


range_clusters = range(2,10)
n_cluster, silhouette_max_value, silhouette_score_values = NumClustersSelection(sim_m_basic, range_clusters)
cluster_labels, centers, silhouette_values, silhouette_avg, labs = ClusterResults(sim_m_basic, n_cluster)
# Outlier
silhouette_min = 0
size_min = 5
out_dist = outlier_distance_detection(sim_m_basic, silhouette_min, size_min, silhouette_values, cluster_labels, n_cluster, centers)

var_exp = graph_kmeans_silhouette(sim_m_basic,labs,n_cluster,cluster_labels,silhouette_values,silhouette_avg)



## DBSCAN
epsilon=1.5
min_samples=4
dbscan_labels,dbscan_n_clusters,dbscan_n_noise,var_exp = dbscan_clustering(sim_m_basic,epsilon,min_samples,None)



## Spectral clustering
range_clusters = range(2,10)
n_cluster, silhouette_max_value, silhouette_score_values = NumClustersSelection(sim_m, range_clusters)
n_neigh = 5
spectral_labels, var_exp = spectral_clustering(sim_m,n_cluster,n_neigh,None)


## Mean shift
bandwidth_value = 1.6
minBinFreq = 2
ms_labels, ms_n_clusters, ms_n_noise, var_exp = MeanShifClustering(sim_m,bandwidth_value,minBinFreq,None)


## Hierarchical
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(sim_m, method='ward'))
plt.show()