from time import time
import numpy as np
import multiprocessing
import pandas as pd
import pprint
from string import punctuation
from .validation import Output
from .tokenization import Tokens
from .ml_clusterization import MLClustering
from .matching_clusterization import SClustering
from .tfidf import TermsAnalysis


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


class Chain(object):

    CLUSTERING_THRESHOLD = 5000
    MATCHING_ACCURACY = 0.8

    def __init__(self, df, target, cluster_settings=None, model_name='word2vec.model', mode='create',
                 threshold=CLUSTERING_THRESHOLD, matching_accuracy=MATCHING_ACCURACY):
        self.df = df
        self.target = target
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.timings = {}
        self.model_name = model_name
        self.mode = mode
        self.threshold = threshold
        self.matching_accuracy = matching_accuracy


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
        self.tokenization()
        self.group_equals()
        if self.groups.shape[0] <= self.CLUSTERING_THRESHOLD:
            self.matching_clusterization(self.groups)
        else:
            self.tokens_vectorization()
            self.sentence_vectorization()
            self.ml_clusterization()
            self.matching_clusterization(self.result)


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        self.tokens = Tokens(self.df[self.target].values)
        self.tokens.process()
        self.df['sequence'] = self.tokens.tokenized_cluster
        self.df['tokenized_pattern'] = self.tokens.tokenized_pattern
        self.tfidf = TermsAnalysis(self.tokens.tokenized_cluster, self.tokens.tokenized_pattern)
        cleaned_tokens = self.tfidf.process()
        self.df['cleaned'] = self.tokens.detokenize(cleaned_tokens)
        print('Tokenization finished')


    @safe_run
    def group_equals(self):

        self.groups = self.df.groupby('cleaned').apply(lambda gr:
                                                pd.DataFrame([{'indices': gr.index.values.tolist(),
                                                              'pattern': gr['cleaned'].values[0],
                                                              'sequence': self.tokens.tokenize_string(
                                                                  self.tokens.TOKENIZER_CLUSTER,
                                                                  gr['cleaned'].values[0],
                                                                  True
                                                              ),
                                                              'tokenized_pattern': self.tokens.tokenize_string(
                                                                  self.tokens.TOKENIZER_PATTERN, gr['cleaned'].values[0]
                                                              ),
                                                               'cluster_size': len(gr.index.values.tolist())}]))
        self.groups.reset_index(drop=True, inplace=True)

        print('Found {} equal groups'.format(self.groups.shape[0]))



    @safe_run
    def tokens_vectorization(self):
        """
        Training word2vec model
        :param iterations:
        :param min_count: minimium frequency count of words (recommended value is 1)
        :return:
        """
        from .vectorization import Vector
        self.vectors = Vector(self.groups['sequence'].values,
                                  self.w2v_size,
                                  self.w2v_window,
                                  self.cpu_number,
                                  self.model_name)
        if self.mode == 'create':
            self.vectors.create_word2vec_model(min_count=1, iterations=10)
        if self.mode == 'update':
            self.vectors.update_word2vec_model()
        if self.mode == 'process':
            self.vectors.load_word2vec_model()
        print('Vectorization of tokens finished')



    @safe_run
    def sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        self.vectors.vectorize_messages()
        print('Vectorization of sentences is finished')
        return self


    @safe_run
    def matching_clusterization(self, groups):
        print('Matching Clusterization!')
        clusters = SClustering(groups, self.tokens, self.matching_accuracy)
        self.result = clusters.matching_clusterization()


    @safe_run
    def ml_clusterization(self):

        self.clusters = MLClustering(self.df, self.groups, self.tokens, self.vectors, self.cpu_number)
        self.result = self.clusters.process()


    def in_cluster(self, groups, cluster_label):
        indices = groups.loc[cluster_label, 'indices']
        return self.df.loc[indices][self.target].values


    @safe_run
    def validation(self, groups):
        return Output().statistics(self.df, self.target, groups)


    def garbage_collector(self, df):
        stop = list(punctuation) + ['｟*｠']
        garbage = []
        for row in df.itertuples():
            elements = set(row.sequence)
            c = 0
            for i,x in enumerate(elements):
                if x in stop:
                    c+=1
            if c == len(elements):
                garbage.append(row)
                print("Founded garbage")
                pprint.pprint(garbage)
                df.drop([row.Index], axis=0, inplace=True)
        return garbage


    @staticmethod
    def split_clusters(df, column, threshold=100):
        if np.max(df[column].values) < threshold:
            return df, None
        else:
            return df[df[column] >= threshold], df[df[column] < threshold]





