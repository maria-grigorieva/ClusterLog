from time import time
import multiprocessing
import pandas as pd
import pprint
from string import punctuation
from .data_preparation import Regex
from .validation import Output
from .tokenization import Tokens
from .clusterization import Clustering


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
        self.timings = {}
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
            .clusterization()


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
        self.df['sequence'] = self.tokens.tokenized_cluster
        self.df['tokenized_pattern'] = self.tokens.tokenized_pattern
        self.df['cleaned'] = self.tokens.patterns
        print('Tokenization finished')
        return self


    @safe_run
    def group_equals(self):

        self.groups = self.df.groupby('cleaned').apply(lambda gr:
                                                pd.DataFrame([{'indices': gr.index.values.tolist(),
                                                              'pattern': gr['cleaned'].values[0],
                                                              'sequence': self.tokens.tokenize_string(
                                                                  self.tokens.TOKENIZER_CLUSTER, gr['cleaned'].values[0]
                                                              ),
                                                              'tokenized_pattern': self.tokens.tokenize_string(
                                                                  self.tokens.TOKENIZER_PATTERN, gr['cleaned'].values[0]
                                                              ),}]))
        self.groups.reset_index(drop=True, inplace=True)

        print('Found {} equal groups'.format(self.groups.shape[0]))

        return self


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
        return self


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
    def clusterization(self):

        self.clusters = Clustering(self.df, self.groups, self.tokens, self.vectors, self.cpu_number)
        self.clusters.process()
        return self


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





