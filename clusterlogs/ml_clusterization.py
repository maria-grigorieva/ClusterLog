from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import math
import numpy as np
import editdistance
from .phraser import phraser
from .LogCluster import *
from .data_preparation import *
from .tokenization import *


class MLClustering:

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method):
        self.groups = groups
        self.df = df
        self.method = method
        self.vectors = vectors
        self.distances = None
        self.epsilon = None
        self.min_samples = 1
        self.cpu_number = cpu_number
        self.add_placeholder = add_placeholder


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
        self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('HDBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))
        return pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)



    def hierarchical(self):
        """
        Agglomerative clusterization
        :return:
        """
        self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=0.1) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)


    def gb_regroup(self, gb):
        # Search for the most common patterns using LogCluster app (Perl)
        pattern = self.logcluster_clusterization(gb['pattern'].values)
        # Generate text from all group sequences
        text = '. '.join([' '.join(row) for row in gb['sequence'].values])
        # Extract common phrases
        phrases_pyTextRank = phraser(text, 'pyTextRank')
        phrases_RAKE = phraser(text, 'RAKE')
        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': pattern,
                'indices': indices,
                'cluster_size': size,
                'common_phrases_pyTextRank': phrases_pyTextRank.extract_common_phrases(),
                'common_phrases_RAKE': phrases_RAKE.extract_common_phrases()}



    def levenshtein_similarity(self, top, rows):
        """
        Search similarities between top and all other sequences of tokens.
        May be used for strings as well.
        top - most frequent sequence
        rows - all sequences
        :param rows:
        :return:
        """
        if len(rows) > 1:
            return (
                [(1 - editdistance.eval(top, rows[i]) / max(len(top), len(rows[i]))) for i in
                 range(0, len(rows))])
        else:
            return [1]



    def drain_clusterization(self, messages):
        regex = [r'(/[\w\./]*[\s]?)', r'([a-zA-Z0-9]+[_]+[\S]+)', r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)', r'[^\w\s]']
        #regex = []
        parser = LogParser(input=messages, st=0.5, rex=regex)
        result = parser.parse()
        return result


    def logcluster_clusterization(self, messages):
        if len(messages) == 1:
            return clean_messages(messages)
        else:
            support = 1 if len(messages) > 1 and len(messages) < 20 else 2
            regex = [r'[^ ]+\.[^ ]+', r'(/[\w\./]*[\s]?)', r'([a-zA-Z0-9]+[_]+[\S]+)', r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)', r'[^\w\s]']
            parser = LogParser(messages=messages, support=support, outdir='',rex=regex)
            patterns = parser.parse()
            if len(patterns) == 0:
                parser = LogParser(messages=messages, support=1, outdir='', rex=regex)
                patterns = parser.parse()
            return patterns