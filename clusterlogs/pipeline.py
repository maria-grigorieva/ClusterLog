import multiprocessing
from math import sqrt
from time import time

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.neighbors import NearestNeighbors
from .tokenization import Tokens
from .data_preparation import Regex
from .cluster_output import Output
from sklearn.decomposition import PCA

import pprint


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


REGEX = [r'[0-9a-zA-Z]{12,128}',
          r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}',
          r'(http[s]|root|srm|file|ftp[s]|hdf[s])*:(//|/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
          r'(/[a-zA-Z\./]*[\s]?)',
          r'(\d+)',
          r'(\b\w+://)\S+(?=\s)',
          r'(\b[f|F]ile( exists)?:?\s?)/\S+(?=\s)',
          r'[-./_a-zA-Z0-9]{25,}']

# REGEX = [r'\d+',
#          r'[0-9a-zA-Z]{12,128}',
#          r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}',
#          ]

CLUSTERING_DEFAULTS = {"w2v_size": "auto",
                       "w2v_window": 7,
                       "min_samples": 1}


class ml_clustering(object):


    def __init__(self, df, target, cluster_settings=None, regex=REGEX, model_name='word2vec.model', mode='create'):
        self.df = df
        self.target = target
        self.regex = regex
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.messages = self.extract_messages()
        self.timings = {}
        self.messages_cleaned = None
        self.tokens = None
        self.sent2vec = None
        self.distances = None
        self.epsilon = None
        self.cluster_labels = None
        self.model_name = model_name
        self.mode = mode
        self.patterns_stats = None
        self.results = None



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
            .dimensionality_reduction() \
            .hdbscan() \
            .extract_patterns() \
            .reprocess() \
            .statistics()
        # return self.data_preparation() \
        #     .tokenization() \
        #     .tokens_vectorization() \
        #     .sentence_vectorization() \
        #     .kneighbors() \
        #     .epsilon_search() \
        #     .dbscan() \
        #     .extract_patterns() \
        #     .reprocess() \
        #     .statistics()

    #.kneighbors() \
        # .epsilon_search() \
    # .dbscan() \

    @safe_run
    def data_preparation(self):
        """
        Cleaning log messages from unnucessary substrings and tokenization
        :return:
        """
        regex = Regex(self.messages, self.regex)
        self.messages_cleaned = regex.process()
        self.df['cleaned'] = self.messages_cleaned
        return self


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        self.tokens = Tokens(self.messages_cleaned)
        self.tokens.process()
        self.df['tokenized_wordpunct'] = self.tokens.tokenized_wordpunct
        self.df['tokenized_pyonmttok'] = self.tokens.tokenized_pyonmttok
        if self.w2v_size == 'auto':
            self.w2v_size = self.detect_embedding_size()
        return self


    def detect_embedding_size(self):
        vocab = self.tokens.get_vocabulary(self.tokens.tokenized_wordpunct)
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
        self.word_vector = Vector(self.tokens.tokenized_wordpunct,
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
    def dimensionality_reduction(self):
        pca = PCA(n_components=20, svd_solver='full')
        pca.fit(self.sent2vec)
        self.sent2vec_PCA = pca.transform(self.sent2vec)
        return self



    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(sqrt(len(self.sent2vec_PCA)))
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nbrs = neigh.fit(self.sent2vec_PCA)
        distances, indices = nbrs.kneighbors(self.sent2vec_PCA)
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
        self.df['cluster_1'] = self.cluster_labels
        return self


    def optics(self):
        from pyclustering.cluster.optics import optics
        optics_instance = optics(self.sent2vec)
        optics_instance.process()
        clusters = optics_instance.get_clusters()


    @safe_run
    def hdbscan(self):
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.sent2vec_PCA)
        self.df['cluster_1'] = self.cluster_labels
        return self


    @safe_run
    def hierarchical(self):
        """
        Agglomerative clusterization
        :return:
        """
        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=self.epsilon)\
            .fit_predict(self.sent2vec)
        self.df['cluster_1'] = self.cluster_labels
        return self


    @safe_run
    def extract_patterns(self):
        """

        :return:
        """
        self.output = Output(self.df, self.target)
        self.patterns_stats = self.output.statistics(output_mode='frame', level=1)
        return self


    @safe_run
    def reprocess(self):
        """

        :return:
        """
        self.output.postprocessing(self.patterns_stats, level=1)
        return self


    @safe_run
    def statistics(self):
        self.results = self.output.statistics(output_mode='frame', level=2, restruct=False)
        return self


    def in_cluster(self, cluster_label, level=1):
        return self.df[self.df['cluster_'+str(level)] == str(cluster_label)][self.target].values