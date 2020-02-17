from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import math
import pandas as pd
import numpy as np
import difflib
from .tokenization import Tokens


class MLClustering:

    def __init__(self, df, groups, tokens, vectors, cpu_number, method='dbscan'):
        self.groups = groups
        self.df = df
        self.method = method
        self.tokens = tokens
        self.vectors = vectors
        self.distances = None
        self.epsilon = None
        self.min_samples = 1
        self.cpu_number = cpu_number


    def process(self):
        if self.method == 'dbscan':
            return self.dbscan()


    def dimensionality_reduction(self):
        n = self.vectors.detect_embedding_size(self.tokens.vocabulary_cluster)
        print('Number of dimensions is {}'.format(n))
        pca = PCA(n_components=n, svd_solver='full')
        pca.fit(self.vectors.sent2vec)
        return pca.transform(self.vectors.sent2vec)


    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(math.sqrt(len(self.vectors.sent2vec)))
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self.vectors.sent2vec)
        distances, indices = nbrs.kneighbors(self.vectors.sent2vec)
        self.distances = [np.mean(d) for d in np.sort(distances, axis=0)]


    def epsilon_search(self):
        """
        Search epsilon for the DBSCAN clusterization
        :return:
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))))
        self.epsilon = max(kneedle.all_elbows) if (len(kneedle.all_elbows) > 0) else 1


    def dbscan(self):
        """
        Execution of the DBSCAN clusterization algorithm.
        Returns cluster labels
        :return:
        """
        self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()
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
        import hdbscan
        self.tokens.sent2vec = self.tokens.sent2vec if self.w2v_size <= 10 else self.dimensionality_reduction()

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.tokens.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)
        print('HDBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))


    def hierarchical(self):
        """
        Agglomerative clusterization
        :return:
        """
        self.tokens.sent2vec = self.tokens.sent2vec if self.w2v_size <= 10 else self.dimensionality_reduction()
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=0.1) \
            .fit_predict(self.tokens.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)


    def gb_regroup(self, gb):
        tokenized_pattern = self.sequence_matcher(gb['tokenized_pattern'].values)
        common_pattern = Tokens.detokenize_row(Tokens.TOKENIZER_PATTERN, tokenized_pattern)
        sequence = self.tokens.tokenize_string(self.tokens.TOKENIZER_PATTERN, common_pattern)
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': common_pattern,
                'sequence': sequence,
                'tokenized_pattern': tokenized_pattern,
                'indices': indices,
                'cluster_size': size}


    def sequence_matcher(self, sequences):
        if len(sequences) > 1:
            pattern = sequences[0]
            for i in range(1,len(sequences)):
                matches = difflib.SequenceMatcher(None, pattern, sequences[i])
                m = [pattern[m.a:m.a + m.size] for m
                          in matches.get_matching_blocks() if m.size > 0]
                pattern = [val for sublist in m for val in sublist]
            return pattern
        else:
            return sequences[0]

