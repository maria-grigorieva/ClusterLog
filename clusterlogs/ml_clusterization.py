import math
import numpy as np
import pandas as pd

from kneed import KneeLocator
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from .phraser import extract_common_phrases
from .tokenization import get_vocabulary, detokenize_messages
from .sequence_matching import Match
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
            return self.dbscan()
        if self.method == 'hdbscan':
            return self.hdbscan()
        if self.method == 'hierarchical':
            return self.hierarchical()

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
        return pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)

    def hdbscan(self):
        self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()

        clusterer = HDBSCAN(min_cluster_size=10, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('HDBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))
        return pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)

    def hierarchical(self):
        """
        Agglomerative clustering
        """
        if len(self.vectors.sent2vec) >= 5000:
            self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 \
                else self.dimensionality_reduction()
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=0.1) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)

    def gb_regroup(self, gb):
        m = Match(gb['tokenized_pattern'].values, add_placeholder=self.add_placeholder)
        tokenized_pattern = []
        sequences = gb['tokenized_pattern'].values

        if len(sequences) > 1:
            m.matching_clusters(sequences, tokenized_pattern)
        elif len(sequences) == 1:
            tokenized_pattern.append(sequences[0])
        pattern = detokenize_messages(tokenized_pattern, self.tokenizer_type)

        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)

        text = '. '.join([' '.join(row) for row in self.df.loc[indices]['sequence'].values])
        phrases = extract_common_phrases(text, 'rake_nltk')

        return {'pattern': pattern,
                'indices': indices,
                'cluster_size': size,
                'common_phrases': phrases}
