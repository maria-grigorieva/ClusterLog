import math
import numpy as np
import pandas as pd

from kneed import KneeLocator
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from .phraser import extract_common_phrases
from .Drain import LogParser
#from .LogCluster import LogParser
from .tokenization import get_vocabulary
from .data_preparation import clean_messages
from .sequence_matching import Match
from .tokenization import detokenize_row
import re

# import editdistance

LIMIT = 30


class MLClustering:

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method, tokenizer_type):
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
        if len(self.vectors.sent2vec) > 10000:
            self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 \
                    else self.dimensionality_reduction()
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

        clusterer = HDBSCAN(min_cluster_size=100, min_samples=1)
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
        # m = Match(gb['pattern'].values)
        # tokenized_pattern = []
        # print(len(gb['tokenized_pattern'].values))
        # m.matching_clusters(gb['tokenized_pattern'].values, tokenized_pattern)
        # pattern = detokenize_messages(tokenized_pattern, self.tokenizer_type)
        # print(len(pattern))
        # Search for the most common patterns using LogCluster app (Perl)
        drain_pattern = self.drain_clusterization(gb['pattern'].values)
        # Generate text from all group sequences
        text = '. '.join([' '.join(row) for row in gb['sequence'].values])
        # Extract common phrases
        #phrases_pyTextRank = Phraser(text, 'pyTextRank')
        phrases_RAKE = extract_common_phrases(text, 'RAKE')
        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': drain_pattern,
                # 'drain_pattern': drain_pattern,
                'indices': indices,
                'cluster_size': size,
                #'common_phrases_pyTextRank': phrases_pyTextRank.extract_common_phrases(),
                'common_phrases_RAKE': phrases_RAKE}


    def drain_clusterization(self, messages):
        #regex = [r'(/[\w\./]*[\s]?)', r'([a-zA-Z0-9]+[_]+[\S]+)', r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)', r'[^\w\s]']
        regex = []
        parser = LogParser(input=messages, rex=regex, st=0.5)
        result = parser.parse()
        cleaned = []
        for line in result:
            cleaned.append(line[0])
        # for line in result:
        #     l = line[0].replace('<*> ', '<*>')
        #     l = re.sub(r'[(<*>)]+', '(.*?)', l)
        #     cleaned.append(l)
        return cleaned


    def logcluster_clusterization(self, messages):
        if len(messages) == 1:
            return clean_messages(messages)
        else:
            support = 1 if len(messages) > 1 and len(messages) < 20 else 2
            #regex = []
            regex = [r'[^ ]+\.[^ ]+', r'(/[\w\./]*[\s]?)', r'([a-zA-Z0-9]+[_]+[\S]+)', r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)', r'[^\w\s]']
            parser = LogParser(messages=messages, support=support, outdir='', rex=regex)
            patterns = parser.parse()
            if len(patterns) == 0:
                parser = LogParser(messages=messages, support=1, outdir='', rex=regex)
                patterns = parser.parse()
            return patterns
