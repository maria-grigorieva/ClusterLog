from time import time
import multiprocessing
import nltk
import numpy as np
import pandas as pd
from itertools import groupby
import pprint
import math
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .tokenization import Tokens
from .data_preparation import Regex
from sklearn.decomposition import PCA
import editdistance
from .validation import Output


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


CLUSTERING_DEFAULTS = {"w2v_size": 300,
                       "w2v_window": 7,
                       "min_samples": 1}


class ml_clustering(object):


    def __init__(self, df, target, cluster_settings=None, model_name='word2vec.model', mode='create'):
        self.df = df
        self.target = target
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
            .tokenization() \
            .group_equals() \
            .tokens_vectorization() \
            .sentence_vectorization() \
            .dbscan()
            # .regroup() \
            # .postprocessing()


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
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        self.tokens = Tokens(self.df['cleaned'].values)
        self.tokens.process()
        self.df['tokenized_dbscan'] = self.tokens.tokenized_dbscan
        self.df['hashed'] = self.tokens.hashed
        self.df['tokenized_pattern'] = self.tokens.tokenized_pattern
        self.df['cleaned'] = self.tokens.patterns
        print('Tokenization finished')
        return self


    @safe_run
    def group_equals(self):

        self.groups = self.df.groupby('cleaned').apply(lambda gr:
                                                pd.DataFrame([{'indices': gr.index.values.tolist(),
                                                              'pattern': gr['cleaned'].values[0],
                                                              'tokenized_dbscan': self.tokens.tokenize_string(
                                                                  self.tokens.tokenizer_dbscan, gr['cleaned'].values[0]
                                                              ),
                                                              'tokenized_pattern': self.tokens.tokenize_string(
                                                                  self.tokens.tokenizer_pattern, gr['cleaned'].values[0]
                                                              ),}]))
        self.groups.reset_index(drop=True, inplace=True)

        print('Found {} equal groups'.format(self.groups.shape[0]))

        return self
    #
    #
    # @safe_run
    # def tokenization(self):
    #     """
    #     Tokenization of a list of error messages.
    #     :return:
    #     """
    #     self.tokens = Tokens(self.groups['pattern'].values)
    #     self.tokens.process()
    #     self.groups['tokenized_dbscan'] = self.tokens.tokenized_dbscan
    #     self.groups['hashed'] = self.tokens.hashed
    #     self.groups['tokenized_pattern'] = self.tokens.tokenized_pattern
    #     print('Tokenization finished')
    #     return self



    def detect_embedding_size(self, vocab):
        """
        Automatic detection of word2vec embedding vector size,
        based on the length of vocabulary.
        Max embedding size = 300
        :return:
        """
        print('Vocabulary size = {}'.format(len(vocab)))
        #embedding_size = round(len(vocab) ** (2/3))
        embedding_size = round(math.sqrt(len(vocab)))
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
        # self.w2v_size = self.detect_embedding_size(self.tokens.vocabulary)
        #tokens = self.tokens.clean_tokens(self.tokens.tokenized)
        self.word_vector = Vector(self.groups['tokenized_dbscan'].values,
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
        n = self.detect_embedding_size(self.tokens.vocabulary_dbscan)
        print('Number of dimensions is {}'.format(n))
        pca = PCA(n_components=n, svd_solver='full')
        pca.fit(self.sent2vec)
        return pca.transform(self.sent2vec)



    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(math.sqrt(len(self.sent2vec)))
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self.sent2vec)
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
        self.sent2vec = self.sent2vec if self.w2v_size <= 10 else self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        #self.result = self.groups.groupby('cluster').apply(func=self.gb_regroup)
        self.result = pd.DataFrame.from_dict([item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
                               orient='columns')
        print('DBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))
        return self



    def gb_regroup(self, gb):
        common_pattern = self.matcher(gb['tokenized_pattern'].values)
        sequence = self.tokens.tokenize_string(self.tokens.tokenizer_pattern, common_pattern)
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': common_pattern,
                'sequence': sequence,
                'indices': indices,
                'cluster_size': size}


    @safe_run
    def regroup(self):

        self.groups_ = pd.DataFrame.from_dict([item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)])

        print('regroup finished')


    @safe_run
    def postprocessing(self, accuracy=CLUSTERING_ACCURACY):

        result = []
        self.reclustering(self.result.copy(deep=True), result, accuracy)

        self.result_pp = pd.DataFrame(result)
        self.result_pp.sort_values(by=['cluster_size'], ascending=False, inplace=True)

        print('postprocessed')


    def reclustering(self, df, result, accuracy):

        df['ratio'] = self.levenshtein_similarity(df['sequence'].values, 0)
        filtered = df[(df['ratio'] >= accuracy)]
        pattern = self.matcher(filtered['sequence'].values)
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        result.append({'pattern':pattern,
                       'indices': indices,
                       'cluster_size': len(indices)})
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result, accuracy)


    def matcher(self, lines):
        if len(lines) > 1:
            fdist = nltk.FreqDist([i for l in lines for i in l])
            x = [token if (fdist[token] / len(lines) >= 1) else '｟*｠' for token in lines[0]]
            return self.tokens.tokenizer_pattern.detokenize([i[0] for i in groupby(x)])
        else:
            return self.tokens.tokenizer_pattern.detokenize(lines[0])


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
                    [(1 - editdistance.eval(rows[0], rows[i]) / (max(len(rows[0]), len(rows[i])) | 1)) for i
                     in
                     range(0, len(rows))])
        else:
            return 1


    def in_cluster(self, groups, cluster_label):
        indices = groups.loc[cluster_label, 'indices']
        return self.df.loc[indices][self.target].values



    def validation(self, groups):
        return Output().statistics(self.df, self.target, groups)


    def split_clusters(self, df, column):
        if np.max(df[column].values) < 100:
            return df, None
        else:
            return df[df[column] >= 100], df[df[column] < 100]




