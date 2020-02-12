from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import math
import pandas as pd
import numpy as np
import difflib
import editdistance
import nltk
from .tokenization import Tokens

CLUSTERING_ACCURACY = 0.8
CLUSTERING_THRESHOLD = 5000

class Clustering:

    def __init__(self, df, groups, tokens, vectors, cpu_number, method='dbscan', threshold=CLUSTERING_THRESHOLD):
        self.groups = groups
        self.df = df
        self.threshold = threshold
        self.method = method
        self.tokens = tokens
        self.vectors = vectors
        self.distances = None
        self.epsilon = None
        self.min_samples = 1
        self.cpu_number = cpu_number


    def process(self):
        if self.groups.shape[0] <= self.threshold:
            self.matching_clusterization(self.groups)
        else:
            if self.method == 'dbscan':
                self.dbscan()


    def dimensionality_reduction(self):
        n = self.vectors.detect_embedding_size(self.tokens.vocabulary_dbscan)
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
        self.tokens.sent2vec = self.tokens.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.tokens.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns')
        print('DBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))


    def matching_clusterization(self, df, accuracy=CLUSTERING_ACCURACY):
        """
        Clusterization messages using sequence matching
        :param df:
        :param accuracy:
        :return:
        """
        result = []
        self.reclustering(df.copy(deep=True), result, accuracy)

        self.result = pd.DataFrame(result)
        self.result.sort_values(by=['cluster_size'], ascending=False, inplace=True)

        print('Postprocessed with {} clusters'.format(self.result.shape[0]))


    def hdbscan(self):
        import hdbscan
        self.tokens.sent2vec = self.tokens.sent2vec if self.w2v_size <= 10 else self.dimensionality_reduction()

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.tokens.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns')
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
            orient='columns')


    def reclustering(self, df, result, accuracy):
        """
        Clusterization of the groups:
        - take the 1st message (pattern) and compare if with others
        - take all messages, which are similar with the 1st with more than 80% and
        join them into the new separate cluster
        - remove these messages from the initial group
        - repeat these steps while group has messages
        :param df:
        :param result:
        :param accuracy:
        :return:
        """
        df['ratio'] = self.levenshtein_similarity(df['pattern'].values)
        filtered = df[(df['ratio'] >= accuracy)]
        pattern = self.sequence_matcher(filtered['tokenized_pattern'].values)
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        result.append({'pattern': pattern,
                       'indices': indices,
                       'cluster_size': len(indices)})
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result, accuracy)


    def gb_regroup(self, gb):
        common_pattern = self.sequence_matcher(gb['tokenized_pattern'].values)
        sequence = self.tokens.tokenize_string(self.tokens.tokenizer_pattern, common_pattern)
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': common_pattern,
                'sequence': sequence,
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
            return Tokens.detokenize_row(Tokens.TOKENIZER_PATTERN, pattern)
        else:
            return Tokens.detokenize_row(Tokens.TOKENIZER_PATTERN, sequences[0])



    def levenshtein_similarity(self, rows):
        """
        :param rows:
        :return:
        """
        if len(rows) > 1:
                return (
                [(1 - editdistance.eval(rows[0], rows[i]) / max(len(rows[0]), len(rows[i]))) for i in
                 range(0, len(rows))])
        else:
            return 1


    def matcher(self, lines):
        if len(lines) > 1:
            fdist = nltk.FreqDist([i for l in lines for i in l])
            x = [token if (fdist[token] / len(lines) >= 1) else '｟*｠' for token in lines[0]]
            return self.tokens.detokenize_row(self.tokens.TOKENIZER_PATTERN, x)
        else:
            self.tokens.detokenize_row(self.tokens.TOKENIZER_PATTERN, lines[0])


    @staticmethod
    def split_clusters(df, column, threshold=100):
        if np.max(df[column].values) < threshold:
            return df, None
        else:
            return df[df[column] >= threshold], df[df[column] < threshold]