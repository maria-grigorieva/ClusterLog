import math
import numpy as np

from kneed import KneeLocator
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

from .tokenization import get_vocabulary
import spacy

nlp = spacy.load("en_core_web_sm")
# LIMIT = 30


class MLClustering:

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method, tokenizer_type, pca):
        self.df = df
        self.groups = groups
        self.vectors = vectors
        self.cpu_number = cpu_number
        self.add_placeholder = add_placeholder
        self.method = method
        self.tokenizer_type = tokenizer_type
        self.pca = pca
        self.min_samples = 1
        self.diversity_factor = 0

    def process(self):
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        # Call a method with the corresponding name
        self.cluster_labels = getattr(self, self.method.lower())()
        self.groups['cluster'] = self.cluster_labels
        print(f"{self.method.title()} clustering finished with {len(set(self.cluster_labels))} clusters")

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
        distances, _ = nbrs.kneighbors(self.vectors.sent2vec)
        return [np.mean(d) for d in np.sort(distances, axis=0)]

    def epsilon_search(self, distances):
        """
        Search epsilon for the DBSCAN clusterization
        """
        kneedle = KneeLocator(distances, list(range(len(distances))), online=True)
        epsilon = np.mean(list(kneedle.all_elbows))
        if epsilon == 0.0:
            epsilon = np.mean(distances)
        self.knee_data = {
            'x': [float(x) for x in kneedle.x],
            'y': [float(y) for y in kneedle.y],
            'knees': [float(x) for x in kneedle.all_elbows],
            'chosen_knee': epsilon
        }
        return epsilon

    def dbscan(self) -> np.ndarray:
        """
        Execution of the DBSCAN clustering algorithm.
        Returns cluster labels
        """
        distances = self.kneighbors()
        epsilon = self.epsilon_search(distances)
        cluster_labels = DBSCAN(eps=epsilon,
                                min_samples=self.min_samples,
                                n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def optics(self) -> np.ndarray:
        """
        Execution of the OPTICS clustering algorithm.
        Returns cluster labels
        """
        cluster_labels = OPTICS(min_samples=2,
                                n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def kmeans(self) -> np.ndarray:
        n_clusters = 30
        model = MiniBatchKMeans(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(self.vectors.sent2vec)
        return cluster_labels  # type: ignore

    def hdbscan(self) -> np.ndarray:
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=self.min_samples)
        cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def hierarchical(self) -> np.ndarray:
        """
        Agglomerative clustering
        """

        # I left the following if as a comment because it was the only different
        # way of calling dimensionality reduction. I am pretty sure we don't need it
        # but I am leaving it up just in case. Delete after confirmation

        # if len(self.vectors.sent2vec) >= 5000:
        #     self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 \
        #         else self.dimensionality_reduction()
        distances = self.kneighbors()
        epsilon = self.epsilon_search(distances)
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity='cosine',
            distance_threshold=epsilon,
            linkage='average'
        )
        cluster_labels = model.fit_predict(self.vectors.sent2vec)
        return cluster_labels
