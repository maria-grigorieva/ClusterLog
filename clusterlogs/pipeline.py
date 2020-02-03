from time import time
import multiprocessing
import editdistance
import nltk
import numpy as np
import difflib
import pandas as pd
from itertools import groupby
import re
from collections import OrderedDict
import pprint
import math
import hashlib
from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
from .tokenization import Tokens
from .data_preparation import Regex
from .cluster_output import Output
from sklearn.decomposition import PCA
import editdistance


CLUSTERING_ACCURACY = 0.8

STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "sequence",
              "mean_similarity",
              "std_similarity",
              "indices"]

def safe_run(method):
    def func_wrapper(self, *args, **kwargs):

        try:
            ts = time()
            result = method(self, *args, **kwargs)
            te = time()
            self.timings[method.__name__] = round((te - ts), 4)
            return result

        except Exception:
            return None

    return func_wrapper


CLUSTERING_DEFAULTS = {"w2v_size": 100,
                       "w2v_window": 7,
                       "min_samples": 1}


class ml_clustering(object):


    def __init__(self, df, target, cluster_settings=None, model_name='word2vec.model', mode='create'):
        self.df = df
        # self.df['cluster'] = 0
        self.target = target
        #self.messages = df[self.target].values
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.messages = None
        self.timings = {}
        self.messages_cleaned = None
        self.indices = None
        self.tokens = None
        self.sent2vec = None
        self.distances = None
        self.epsilon = None
        self.cluster_labels = None
        self.model_name = model_name
        self.mode = mode
        self.patterns_stats = None
        self.results = None
        self.groups = None


    @staticmethod
    def get_cpu_number():
        return multiprocessing.cpu_count()


    def set_cluster_settings(self, params):
        for key,value in CLUSTERING_DEFAULTS.items():
            if params.get(key) is not None:
                setattr(self, key, params.get(key))
            else:
                setattr(self, key, value)


    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        :return:
        """
        return self.data_preparation() \
            .group_equals() \
            .tokenization() \
            .tokens_vectorization() \
            .sentence_vectorization() \
            .dbscan() \
            .regroup() \
            .postprocessing()
            # .group_equals() \
            # .split_clusters()


    @safe_run
    def data_preparation(self):
        """
        Cleaning log messages from unnucessary substrings and tokenization
        :return:
        """
        self.preprocessed = Regex(self.df[self.target].values)
        self.df['cleaned'] = self.preprocessed.process()
        print('Data Preparation finished')
        return self


    @safe_run
    def group_equals(self):

        gp = self.df.groupby('cleaned')
        self.messages = [i for i in gp.groups]
        groups = []
        for key, value in gp:
            indices = value.index.values.tolist()
            pattern = value['cleaned'].values[0]
            groups.append({'pattern':pattern, 'indices':indices})
        # arr_slice = np.array(self.df[['cleaned']].values)
        # unq, unqtags, counts = np.unique(arr_slice, return_inverse=True, return_counts=True)
        #self.indices = self.df.index.values
        #self.messages_cleaned = unq
        self.groups = pd.DataFrame(groups)
        print('group_equals finished')

        return self


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        self.tokens = Tokens(self.groups['pattern'].values)
        self.tokens.process()
        self.groups['tokenized'] = self.tokens.tokenized
        #self.df['tokenized'] = self.tokens.tokenized
        # self.df['tokenized_cleaned'] = self.tokens.tokenized_cleaned
        print('Tokenization finished')
        return self



    def detect_embedding_size(self, vocab):
        """
        Automatic detection of word2vec embedding vector size,
        based on the length of vocabulary.
        Max embedding size = 300
        :return:
        """
        embedding_size = round(len(vocab) ** (2/3))
        if embedding_size >= 300:
            embedding_size = 300
        return embedding_size


    @safe_run
    def tokens_vectorization(self):
        """
        Training word2vec model
        :param iterations:
        :param min_count: minimium frequency count of words (recommended value is 1)
        :return:
        """
        from .vectorization import Vector
        self.w2v_size = self.detect_embedding_size(self.tokens.vocabulary)
        #tokens = self.tokens.clean_tokens(self.tokens.tokenized)
        self.word_vector = Vector(self.tokens.tokenized,
                                  self.w2v_size,
                                  self.w2v_window,
                                  self.cpu_number,
                                  self.model_name)
        if self.mode == 'create':
            self.word_vector.create_word2vec_model(min_count=1, iterations=10)
        if self.mode == 'update':
            self.word_vector.update_word2vec_model()
        if self.mode == 'process':
            self.word_vector.load_word2vec_model()
        print('Vectorization of tokens finished')
        return self


    @safe_run
    def sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        self.sent2vec = self.word_vector.sent2vec()
        print('Vectorization of sentences is finished')
        return self


    @safe_run
    def dimensionality_reduction(self):
        pca = PCA(n_components=10, svd_solver='full')
        pca.fit(self.sent2vec)
        return pca.transform(self.sent2vec)



    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        X = self.sent2vec
        k = round(math.sqrt(len(X)))
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        self.distances = [np.mean(d) for d in np.sort(distances, axis=0)]
        return self


    @safe_run
    def epsilon_search(self):
        """
        Search epsilon for the DBSCAN clusterization
        :return:
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))))
        self.epsilon = max(kneedle.all_elbows) if (len(kneedle.all_elbows) > 0) else 1
        return self


    @safe_run
    def dbscan(self):
        """
        Execution of the DBSCAN clusterization algorithm.
        Returns cluster labels
        :return:
        """
        self.sent2vec = self.sent2vec if self.w2v_size <= 10 else self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('DBSCAN finished')
        return self


    @safe_run
    def regroup(self):
        gb = self.groups.groupby('cluster')
        groups = []
        for key,value in gb:
            patterns = value['tokenized'].values
            common_pattern = self.matcher(patterns)
            indices = [i for sublist in value['indices'].values for i in sublist]
            groups.append({'pattern': common_pattern,
                           'sequence': self.tokens.tokenize_string(common_pattern),
                            'indices': indices})
        self.groups = pd.DataFrame(groups)
        print('regroup finished')
        return self


    def postprocessing(self):

        result = {}
        self.reclustering(self.groups.copy(deep=True), result)

        self.df['cluster'] = -1
        self.df['pattern'] = ''
        start = 0
        for key,value in result.items():
            self.df.loc[value, 'cluster'] = start
            self.df.loc[value, 'pattern'] = key
            start += 1

        self.output = Output(self.df, self.target)
        self.result = self.output.statistics()
        self.output.split_clusters(self.result)
        print('postprocessed')
        #self.statistics()
        return self


    def reclustering(self, df, result):

        df['ratio'] = self.levenshtein_similarity(df['sequence'].values, 0)
        filtered = df[(df['ratio'] >= CLUSTERING_ACCURACY)]
        pattern = self.matcher(filtered['sequence'].values)
        result[pattern] = [item for sublist in filtered['indices'].values for item in sublist]
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)


    def matcher(self, lines):
        if len(lines) > 1:
            fdist = nltk.FreqDist([i for l in lines for i in l])
            x = [token if (fdist[token] / len(lines) >= 1) else '{*}' for token in lines[0]]
            return self.tokens.tokenizer.detokenize([i[0] for i in groupby(x)])
        else:
            return self.tokens.tokenizer.detokenize(lines[0])



    def levenshtein_similarity(self, rows, N):
        """
        :param rows:
        :return:
        """
        if len(rows) > 1:
            if N != 0:
                return (
                [(1 - editdistance.eval(rows[0][:N], rows[i][:N]) / max(len(rows[0][:N]), len(rows[i][:N]))) for i in
                 range(0, len(rows))])
            else:
                return (
                    [(1 - editdistance.eval(rows[0], rows[i]) / max(len(rows[0]), len(rows[i]))) for i
                     in
                     range(0, len(rows))])
        else:
            return 1


    def output(self):
        self.df['cluster'] = -1
        start = 0
        for key,value in self.groups:
            indices = value.index.values.tolist()
            self.df.loc[indices, 'cluster'] = start
            start += 1



    def in_cluster(self, cluster_label):
        return self.df[self.df['cluster'] == cluster_label][self.target].values


