import math
import numpy as np
# import pandas as pd

from kneed import KneeLocator
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

from .tokenization import get_vocabulary
import spacy

nlp = spacy.load("en_core_web_sm")
LIMIT = 30


class MLClustering:

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method, tokenizer_type, pca):
        self.groups = groups
        self.df = df
        self.method = method
        self.vectors = vectors
        self.distances = None
        self.epsilon = None
        self.min_samples = 1
        self.cpu_number = cpu_number
        self.add_placeholder = add_placeholder
        self.tokenizer_type = tokenizer_type
        self.diversity_factor = 0
        self.pca = pca

    def process(self):
        if self.method == 'dbscan':
            self.dbscan()
        if self.method == 'hdbscan':
            self.hdbscan()
        if self.method == 'hierarchical':
            self.hierarchical()
        if self.method == 'optics':
            self.optics()
        if self.method == 'kmeans':
            self.kmeans()

    def dimensionality_reduction(self):
        n = self.vectors.detect_embedding_size(get_vocabulary(self.groups['sequence']))
        print('Number of dimensions is {}'.format(n))
        pca = PCA(n_components=n, svd_solver='full')
        pca.fit(self.vectors.sent2vec)
        return pca.transform(self.vectors.sent2vec)

    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        """
        k = round(math.sqrt(len(self.vectors.sent2vec)))
        print('K-neighbours = {}'.format(k))
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='auto').fit(self.vectors.sent2vec)
        distances, indices = nbrs.kneighbors(self.vectors.sent2vec)
        self.distances = [np.mean(d) for d in np.sort(distances, axis=0)]

    def epsilon_search(self):
        """
        Search epsilon for the DBSCAN clusterization
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))), online=True)
        self.epsilon = np.mean(list(kneedle.all_elbows))
        if self.epsilon == 0.0:
            self.epsilon = np.mean(self.distances)
        self.knee_data = {
            'x': [float(x) for x in kneedle.x],
            'y': [float(y) for y in kneedle.y],
            'knees': [float(x) for x in kneedle.all_elbows],
            'chosen_knee': self.epsilon
        }

    def dbscan(self):
        """
        Execution of the DBSCAN clustering algorithm.
        Returns cluster labels
        """
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('DBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))

    def optics(self):
        """
        Execution of the OPTICS clustering algorithm.
        Returns cluster labels
        """
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        self.cluster_labels = OPTICS(min_samples=2,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('OPTICS finished with {} clusters'.format(len(set(self.cluster_labels))))

    def kmeans(self):
        n_clusters = 30
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        model = MiniBatchKMeans(n_clusters=n_clusters)
        self.cluster_labels = model.fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('k-means finished with {} clusters'.format(n_clusters))

    def hdbscan(self):
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=self.min_samples)
        self.cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('HDBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))

    def hierarchical(self):
        """
        Agglomerative clustering
        """
        if len(self.vectors.sent2vec) >= 5000:
            self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 \
                else self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      affinity='cosine',
                                                      distance_threshold=self.epsilon,
                                                      linkage='average') \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('Hierarchical finished with {} clusters'.format(len(set(self.cluster_labels))))
