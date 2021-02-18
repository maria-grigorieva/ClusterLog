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

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method, tokenizer_type, pca, parameters):
        self.df = df
        self.groups = groups
        self.vectors = vectors
        self.cpu_number = cpu_number
        self.add_placeholder = add_placeholder
        self.method = method
        self.tokenizer_type = tokenizer_type
        self.pca = pca
        self.parameters = parameters
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

    def kneighbors(self, metric='euclidean'):
        """
        Calculates average distances for k-nearest neighbors
        """
        k = round(math.sqrt(len(self.vectors.sent2vec)))
        print('K-neighbours = {}'.format(k))
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1, algorithm='auto').fit(self.vectors.sent2vec)
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
            'knees': [float(x) for x in kneedle.all_elbows]
        }
        return float(epsilon)

    def dbscan(self) -> np.ndarray:
        """
        Execution of the DBSCAN clustering algorithm.
        Returns cluster labels
        """
        parameters = {
            'epsilon': None,
            'metric': 'euclidean',
            'min_samples': 1
        }
        parameters.update(self.parameters)

        distances = self.kneighbors(metric=parameters['metric'])
        epsilon = self.epsilon_search(distances)
        if parameters['epsilon'] is None:
            parameters['epsilon'] = epsilon
        self.knee_data['chosen_knee'] = parameters['epsilon']

        cluster_labels = DBSCAN(eps=parameters['epsilon'],
                                metric=parameters['metric'],
                                min_samples=parameters['min_samples'],
                                n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def optics(self) -> np.ndarray:
        """
        Execution of the OPTICS clustering algorithm.
        Returns cluster labels
        """
        parameters = {
            'metric': 'euclidean',
            'min_samples': 2
        }
        parameters.update(self.parameters)

        cluster_labels = OPTICS(min_samples=max(2, parameters['min_samples']),  # type: ignore
                                metric=parameters['metric'],
                                n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def kmeans(self) -> np.ndarray:
        parameters = {
            'n': 30
        }
        parameters.update(self.parameters)

        model = MiniBatchKMeans(n_clusters=parameters['n'])
        cluster_labels = model.fit_predict(self.vectors.sent2vec)
        return cluster_labels  # type: ignore

    def hdbscan(self) -> np.ndarray:
        parameters = {
            'metric': 'euclidean',
            'min_samples': 1
        }
        parameters.update(self.parameters)

        clusterer = HDBSCAN(
            min_cluster_size=max(parameters['min_samples'], 2),  # type: ignore
            min_samples=parameters['min_samples']
        )
        cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        return cluster_labels

    def hierarchical(self) -> np.ndarray:
        """
        Agglomerative clustering
        """
        parameters = {
            'epsilon': None,
            'metric': 'euclidean'
        }
        parameters.update(self.parameters)

        if parameters['epsilon'] is None:
            distances = self.kneighbors(metric=parameters['metric'])
            parameters['epsilon'] = self.epsilon_search(distances)
        linkage = 'ward' if parameters['metric'] == 'euclidean' else 'complete'

        model = AgglomerativeClustering(
            n_clusters=None,
            affinity=parameters['metric'],
            distance_threshold=parameters['epsilon'],
            linkage=linkage
        )
        cluster_labels = model.fit_predict(self.vectors.sent2vec)
        return cluster_labels
