import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
import scipy
import copy
from sklearn import metrics
import pandas as pd
from sklearn.cluster import DBSCAN




################################################################################################
###########                             FUNCTIONS DBSCAN                             ###########
################################################################################################

def DBSCAN_parameters_epsilon_minsamples(data):
    ## fake is inside data
    ## Selecting only the news in the knowledge based graph
    data=data[0:-1, 0:-1]
    ## Descomposition
    u, s, v = np.linalg.svd(data)
    ## Number of eigen values greater than 1
    num_eigenvalues=len(s[s>1])

    ## In case that num_eigenvalues is less than 5, we change it to 9 to avoid problems in the loop of min_samples
    if (num_eigenvalues<=5):
        num_eigenvalues=9

    ## Number of news
    len_news=data.shape[0]

    ## Upper and lower threshols for number of noisy points
    #threshold1=np.floor(len_news*0.05)
    threshold2=np.ceil(len_news*0.05)
    error=0
    flag = 0
    for i in range(20,90):
        if flag==1:
            break
        epsilon = i/100
        for min_samples in range(num_eigenvalues,(5-1),-1):
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed').fit(data)
            dbscan_n_noise = list(dbscan.labels_).count(-1)

            if (dbscan_n_noise <= threshold2):
                #((dbscan_n_noise >= threshold1) and(dbscan_n_noise <= threshold2))
                flag = 1
                break
    if (flag==0):
        error = 1
        print("There are not values of epsilon and min_samples values that meet the Sensitivity condition")
    return epsilon, min_samples, error


def prob_noise_points(data, epsilon, min_samples, dbscan_labels):
    # data is filtered knowledge graphic + fake_news
    # epsilon used for dbscan
    # min_samples used for dbscan
    # dbscan_labels

    dis_matrix = data[0:-1, 0:-1]
    noise_points = dis_matrix[(dbscan_labels == -1),:]
    noise_cluster = []
    # dataframe to save prob of fake
    noise_index = pd.DataFrame(np.argwhere(dbscan_labels == -1),columns=["News"])
    noise_index["Prob_fake"] = None
    noise_index["Error"] = None


    num_clust = np.unique(dbscan_labels[dbscan_labels >= 0])
    if len(num_clust) == 0: # all points are noise points
        noise_index.iloc[:, 1] = 1 #prob_fake=1 for all points
    else: # there is at least one cluster in the knowledge graph
        for j in range(len(noise_index)):
            noise = noise_points[j]
            for i in num_clust:
                dist_noise_clust_i = noise[(dbscan_labels == i)]
                cand = dist_noise_clust_i[dist_noise_clust_i <= epsilon]
                cand_num = np.sum(dist_noise_clust_i <= epsilon)
                noise_cluster.append([i, cand_num, np.sum(cand)])

            # Number of points inside epsilon
            max_points_eps = max([noise_cluster[x][1] for x in range(len(noise_cluster))])
            prob_fake = 1 - (max_points_eps / min_samples)
            noise_index.iloc[j, 1] = prob_fake
            if max_points_eps >= min_samples:
                noise_index.iloc[j, 2] = "ALERTA"
    return noise_index



def dbscan_predict(data, epsilon, min_samples, dbscan_labels):
    # fake is inside data
    # epsilon used for dbscan
    # min_samples used for dbscan
    # dbscan_labels

    #dis_matrix = data[0:-1, 0:-1]
    fake = data[-1][:-1]
    fake_cluster = []  # empty list for cluster assignment
    fake_noise = []  # empty list for noise point and probability

    num_clust = np.unique(dbscan_labels[dbscan_labels >= 0])

    if len(num_clust) == 0:  # all points are noise points
        label_fake = -1  # fake news is a noise point
        max_points_eps = 0 # to make prob_fake = 1
    else:  # there is at least one cluster in the knowledge graph
        for i in num_clust:
            # print(i)
            dist_fake_clust_i = fake[(dbscan_labels == i)]
            cand = dist_fake_clust_i[dist_fake_clust_i <= epsilon]
            cand_num = np.sum(dist_fake_clust_i <= epsilon)
            if cand_num >= min_samples:
                fake_cluster.append([i, cand_num, np.sum(cand)])  # cluster, num points in eps, dist to that points
            else:
                fake_noise.append([i, cand_num, np.sum(cand)])

        if not fake_cluster:  # if fake_cluster is empty
            label_fake = -1  # fake news is a noise point
            # We calculate probability of fake
            max_points_eps = max([fake_noise[x][1] for x in range(len(fake_noise))])

        else:  # if fake_cluster is not empty
            max_points_eps = min_samples  # to make prob_fake = 0
            # From all potential cluster we select that cluster with more closer points to the fake
            max_points_pos = np.argmax([fake_cluster[x][1] for x in range(len(fake_cluster))])

            # If max_points_pos is equal for several cluster, we check distance
            # to assign fake to its closer cluster
            list_cand = [max_points_pos]
            for j in range(len(fake_cluster)):
                if (j!=max_points_pos) and (fake_cluster[j][1]==fake_cluster[max_points_pos][1]):
                    list_cand.append(j)
            if len(list_cand) > 1: # more than one cluster, we look for minimum distance
                pos_aux = np.argmin([fake_cluster[x][2] for x in list_cand])
                max_points_pos = list_cand[pos_aux]

            label_fake = fake_cluster[max_points_pos][0]  # fake news belongs to cluster min_dist_pos

    prob_fake = 1 - (max_points_eps / min_samples)

    return label_fake, prob_fake


def dbscan_clustering(data, epsilon, min_samples, graph_labels, graph_y_n):
    # DBSCAN algorithm requires 2 parameters (DBSCAN doesn't have centers):
    # epsilon , which specifies how close points should be to each other to be considered a part of a cluster;
    # and min_samples , which specifies how many neighbors a point should have to be included into a cluster.

    # data is a distance function with fake news included
    # if graph_labels="all", graph will have labels for all points, otherwise only for noise points

    # We split data in knowledge graph and fake news
    dis_matrix = data[0:-1, 0:-1]
    fake = data[-1][:-1]

    # We compute DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed').fit(dis_matrix)
    dbscan_labels = dbscan.labels_
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_n_noise = list(dbscan_labels).count(-1)

    # We predict new point
    label_fake, prob_fake = dbscan_predict(data, epsilon, min_samples, dbscan_labels)

    if label_fake == -1:
        dbscan_n_noise += 1

    # We add label_fake to dbscan_labels
    dbscan_labels_fake = np.append(dbscan_labels, label_fake)
    var_exp = []

    if graph_y_n == True:
        # Graph
        pca = PCA(n_components=3)
        pca_2d = pca.fit_transform(dis_matrix)
        var_exp = pca.explained_variance_ratio_
        pca_fake = pca.transform(fake.reshape(1, -1))  # pca fake
        pca_fake_df = pd.DataFrame(pca_fake, columns=['PC1', 'PC2', 'PC3'])
        pca_fake_df["News"] = "fake"
        pca_fake_df.set_index("News", inplace=True)

        graph_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2', 'PC3'])
        graph_df['News'] = graph_df.index
        graph_df.set_index("News", inplace=True)
        graph_df = graph_df.append(pca_fake_df)

        # We split labels in two: labels of clustered data and labels of noise points
        labels_clustered = dbscan_labels_fake[dbscan_labels_fake >= 0]
        # labels_noise = dbscan_labels[dbscan_labels < 0]
        # For labels
        news = list(graph_df.index)

        if graph_labels == "all":  # label for all points
            # PCA 1 and 2
            plt.scatter(graph_df.iloc[dbscan_labels_fake >= 0, 0], graph_df.iloc[dbscan_labels_fake >= 0, 1],
                        c=labels_clustered, s=50,
                        cmap='viridis')
            plt.scatter(graph_df.iloc[dbscan_labels_fake < 0, 0], graph_df.iloc[dbscan_labels_fake < 0, 1], c='r',
                        marker='*', s=50)
            for i, txt in enumerate(news):
                plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
            # We add fake news
            # plt.scatter(pca_fake[0][0],pca_fake[0][1],c="r", s=50)
            # plt.annotate("fake", (pca_fake[0][0],pca_fake[0][1]))
            plt.title("DBSCAN: Visualization of the clustered data")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.show()

            # PCA 1 and 3
            plt.scatter(graph_df.iloc[dbscan_labels_fake >= 0, 0], graph_df.iloc[dbscan_labels_fake >= 0, 2],
                        c=labels_clustered, s=50,
                        cmap='viridis')
            plt.scatter(graph_df.iloc[dbscan_labels_fake < 0, 0], graph_df.iloc[dbscan_labels_fake < 0, 2], c='r',
                        marker='*', s=50)
            for i, txt in enumerate(news):
                plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 2]))
            # We add fake news
            # plt.scatter(pca_fake[0][0],pca_fake[0][1],c="r", s=50)
            # plt.annotate("fake", (pca_fake[0][0],pca_fake[0][1]))
            plt.title("DBSCAN: Visualization of the clustered data")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 3")
            plt.show()
        else:  # label for noise points
            # PCA 1 and PCA 2
            plt.scatter(graph_df.iloc[dbscan_labels_fake >= 0, 0], graph_df.iloc[dbscan_labels_fake >= 0, 1],
                        c=labels_clustered, s=50, cmap='viridis')
            plt.scatter(graph_df.iloc[dbscan_labels_fake < 0, 0], graph_df.iloc[dbscan_labels_fake < 0, 1], c='r',
                        marker='*', s=50)
            for i, txt in enumerate(news):
                if (dbscan_labels_fake[i] == -1) or (graph_df.index[i] == "fake"):
                    plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 1]))
            # plt.scatter(pca_fake[0][0], pca_fake[0][1], c="r", s=50)
            # plt.annotate("fake", (pca_fake[0][0], pca_fake[0][1]))
            plt.title("DBSCAN: Visualization of the clustered data")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.show()

            # PCA 1 and PCA 3
            plt.scatter(graph_df.iloc[dbscan_labels_fake >= 0, 0], graph_df.iloc[dbscan_labels_fake >= 0, 2],
                        c=labels_clustered, s=50, cmap='viridis')
            plt.scatter(graph_df.iloc[dbscan_labels_fake < 0, 0], graph_df.iloc[dbscan_labels_fake < 0, 2], c='r',
                        marker='*', s=50)
            for i, txt in enumerate(news):
                if (dbscan_labels_fake[i] == -1) or (graph_df.index[i] == "fake"):
                    plt.annotate(txt, (graph_df.iloc[i, 0], graph_df.iloc[i, 2]))
            # plt.scatter(pca_fake[0][0], pca_fake[0][1], c="r", s=50)
            # plt.annotate("fake", (pca_fake[0][0], pca_fake[0][1]))
            plt.title("DBSCAN: Visualization of the clustered data")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 3")
            plt.show()

    return dbscan_labels, dbscan_n_clusters, dbscan_n_noise, var_exp, label_fake, prob_fake


