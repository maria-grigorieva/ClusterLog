import multiprocessing
from math import sqrt
from time import time

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from .helper import *
from .tokenization import Tokens


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


CLUSTERING_DEFAULTS = {"tokenizer": "nltk",
                       "w2v_size": "auto",
                       "w2v_window": 7,
                       "min_samples": 1}


class ml_clustering:


    def __init__(self, df, target, cluster_settings=None, model_name='word2vec.model', mode='create'):
        self.df = df
        self.target = target
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.messages = self.extract_messages()
        self.timings = {}
        self.messages_cleaned = None
        self.tokenized = None
        self.sent2vec = None
        self.distances = None
        self.epsilon = None
        self.cluster_labels = None
        self.model_name = model_name
        self.mode = mode


    @staticmethod
    def get_cpu_number():
        return multiprocessing.cpu_count()


    def set_cluster_settings(self, params):
        for key,value in CLUSTERING_DEFAULTS.items():
            if params.get(key) is not None:
                setattr(self, key, params.get(key))
            else:
                setattr(self, key, value)


    def extract_messages(self):
        """
        Returns a list of all error messages from target column
        :return:
        """
        return list(self.df[self.target])


    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        :return:
        """
        return self.data_preparation() \
            .tokenization() \
            .tokens_vectorization() \
            .sentence_vectorization() \
            .kneighbors() \
            .epsilon_search() \
            .dbscan()

    def reprocess(self, epsilon):
        self.epsilon = epsilon
        return self.dbscan()

    @safe_run
    def data_preparation(self):
        """
        Cleaning log messages from unnucessary substrings and tokenization
        :return:
        """
        self.messages_cleaned = cleaner(self.messages)
        self.df['_cleaned'] = self.messages_cleaned
        return self


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        tokens = Tokens(self.messages_cleaned, type=self.tokenizer)
        self.tokenized = tokens.process()
        if self.w2v_size == 'auto':
            self.w2v_size = self.detect_embedding_size(tokens)
        return self


    def detect_embedding_size(self, tokens):
        vocab = tokens.get_vocabulary()
        embedding_size = round(len(vocab) ** (2/3))
        if embedding_size >= 400:
            embedding_size = 400
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
        self.word_vector = Vector(self.tokenized,
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
        return self


    @safe_run
    def sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        self.sent2vec = self.word_vector.sent2vec()
        return self


    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(sqrt(len(self.sent2vec)))
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(self.sent2vec)
        distances, indices = nbrs.kneighbors(self.sent2vec)
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
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.sent2vec)
        return self


    def hierarchical(self):
        """
        Agglomerative clusterization
        :return:
        """
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=self.epsilon)\
            .fit_predict(self.sent2vec)
        return self