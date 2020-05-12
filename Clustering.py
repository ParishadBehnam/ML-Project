import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


class Cluster:

    def __init__(self):
        self.testing_X = pd.read_csv('../data/Test/X_test.txt', sep=" ", header=None)
        self.testing_y = pd.read_csv('../data/Test/y_test.txt', sep=" ", header=None)
        self.testing_si = pd.read_csv('../data/Test/subject_id_test.txt', sep=" ", header=None)

        self.training_X = pd.read_csv('../data/Train/X_train.txt', sep=" ", header=None)
        self.training_y = pd.read_csv('../data/Train/y_train.txt', sep=" ", header=None)
        self.training_si = pd.read_csv('../data/Train/subject_id_train.txt', sep=" ", header=None)

    def kmeans_fit(self):
        best_cost = np.inf
        best_kmeans = None
        # for alg in ['full', 'elkan']:
        for _ in range(50):
            kmeans = KMeans(n_clusters=12, precompute_distances='auto').fit(self.training_X)
            cost = kmeans.inertia_
            # print(cost)
            if cost < best_cost:
                best_cost = cost
                best_kmeans = kmeans

        self.kmeans_cluster = best_kmeans

    def kmeans_predict(self):
        return self.kmeans_cluster.predict(self.testing_X)

    def kmeans_rand_index(self):
        y_hat = self.kmeans_predict()
        print('K-means Rand Index:')
        print(metrics.adjusted_rand_score(np.array(self.testing_y).reshape(y_hat.shape[0]), y_hat))
        return y_hat

    def gmm_fit(self):
        best_cost = np.inf
        best_gmm = None
        covs = ['full']  # 'tied', 'diag', 'spherical'
        inits = ['kmeans']  # , 'random'
        # for cov in covs:
        #     for init in inits:
        for _ in range(50):
            gmm = GaussianMixture(n_components=12, covariance_type='full', init_params='kmeans', n_init=3,
                                  max_iter=20).fit(self.training_X)
            cost = -gmm.lower_bound_
            if cost < best_cost:
                best_gmm = gmm
                best_cost = cost
                # print(cost)
        self.gmm_cluster = best_gmm

    def gmm_predict(self):
        return self.gmm_cluster.predict(self.testing_X)

    def gmm_rand_index(self):
        y_hat = self.gmm_predict()
        print('GMM Rand Index:')
        print(metrics.adjusted_rand_score(np.array(self.testing_y).reshape(y_hat.shape[0]), y_hat))
        return y_hat

    def reduce_dimensionality(self, components=20):
        pca = PCA(n_components=components)
        self.training_X = pca.fit_transform(self.training_X)
        self.testing_X = pca.transform(self.testing_X)

    def plot(self, gmm_y, kmeans_y):
        colors = ['b', 'r', 'orange', 'g', 'c', 'pink', 'gray', 'brown', 'black', 'm', 'y', 'violet']
        X = np.array(self.training_X)
        plt.subplot(1, 2, 1)
        for c in range(12):
            c_class = np.where(gmm_y == (c + 1))
            plt.scatter(X[c_class, 0], X[c_class, 1], color=colors[c], s=1, label='Cluster ' + str(c + 1))
        plt.legend()

        plt.subplot(1, 2, 2)
        for c in range(12):
            c_class = np.where(kmeans_y == (c + 1))
            plt.scatter(X[c_class, 0], X[c_class, 1], color=colors[c], s=1, label='Cluster ' + str(c + 1))
        plt.legend()

        plt.show()


cluster = Cluster()


def q_1():
    cluster.gmm_fit()
    gmm_y = cluster.gmm_rand_index()

    cluster.kmeans_fit()
    kmeans_y = cluster.kmeans_rand_index()

    return (gmm_y, kmeans_y)


def q_2():
    cluster.reduce_dimensionality()
    (g, k) = q_1()
    # cluster.reduce_dimensionality(2)
    cluster.plot(g, k)


q_2()
