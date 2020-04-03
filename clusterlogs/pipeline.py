from time import time
import numpy as np
import multiprocessing
import pandas as pd
import pprint
from string import punctuation
from .validation import Output
from .tokenization import *
from .ml_clusterization import MLClustering
from .similarity_clusterization import SClustering
from .data_preparation import *
import hashlib
from .sequence_matching import Match
from .reporting import report


def safe_run(method):
    def func_wrapper(self, *args, **kwargs):

        try:
            ts = time()
            result = method(self, *args, **kwargs)
            te = time()
            self.timings[method.__name__] = round((te - ts), 4)
            return result

        except Exception as e:
            print(e)

    return func_wrapper


CLUSTERING_DEFAULTS = {"w2v_size": 300,
                       "w2v_window": 7,
                       "min_samples": 1}


class Chain(object):

    CLUSTERING_THRESHOLD = 5000
    MATCHING_ACCURACY = 0.8
    CLUSTERING_TYPE = 'SIMILARITY'
    ALGORITHM = 'dbscan'

    def __init__(self, df, target,
                 tokenizer_type='space',
                 cluster_settings=None,
                 model_name='word2vec.model',
                 mode='create',
                 output_file='report.html',
                 add_placeholder=False,
                 threshold=CLUSTERING_THRESHOLD,
                 matching_accuracy=MATCHING_ACCURACY,
                 clustering_type=CLUSTERING_TYPE,
                 algorithm=ALGORITHM):
        self.df = df
        self.target = target
        self.tokenizer_type = tokenizer_type
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.timings = {}
        self.model_name = model_name
        self.mode = mode
        self.threshold = threshold
        self.matching_accuracy = matching_accuracy
        self.clustering_type = clustering_type
        self.add_placeholder = add_placeholder
        self.algorithm = algorithm
        self.output_file = output_file


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
        self.df['tokenized_pattern'] = tokenize_messages(self.df[self.target].values, self.tokenizer_type)

        cleaned_strings = clean_messages(self.df[self.target].values)
        cleaned_tokens = [row.split(' ') for row in cleaned_strings]
        # get frequence of cleaned tokens
        frequency = get_term_frequencies(cleaned_tokens)
        # remove tokens that appear only once and save tokens which are textual substrings
        cleaned_tokens = [
            [token for token in row if frequency[token] > 1]
            for row in cleaned_tokens]
        cleaned_strings = [' '.join(row) for row in cleaned_tokens]
        self.df['hash'] = self.generateHash(cleaned_strings)

        self.df['sequence'] = cleaned_tokens

        self.group_equals(self.df, 'hash')

        if self.clustering_type == 'SIMILARITY' and self.groups.shape[0] <= self.CLUSTERING_THRESHOLD:
                clusters = SClustering(self.groups, self.matching_accuracy, self.add_placeholder, self.tokenizer_type)
                self.result = clusters.process()
                print('Finished with {} clusters'.format(self.result.shape[0]))
        else:
            self.tokens_vectorization()
            self.sentence_vectorization()
            self.ml_clusterization()

        report.generate_html_report(self.result, self.output_file)



    def generateHash(self, sequences):
        return [hashlib.md5(repr(row).encode('utf-8')).hexdigest() for row in sequences]


    # @safe_run
    # def tfidf(self):
    #     """
    #     Generate TF-IDF model and remove tokens with max weights
    #     :return:
    #     """
    #     self.tfidf = TermsAnalysis(self.tokens)
    #     cleaned_tokens = self.tfidf.process()
    #     print('Tokens TF-IDF cleaning finished')
    #     return cleaned_tokens


    @safe_run
    def group_equals(self, df, column):

        self.groups = df.groupby(column).apply(func=self.regroup)
        self.groups.reset_index(drop=True, inplace=True)

        print('Found {} equal groups'.format(self.groups.shape[0]))


    @safe_run
    def regroup(self, gr):
        """
        tokenized_pattern - common sequence of tokens, generated based on all tokens
        sequence - common sequence of tokens, based on cleaned tokens
        pattern - textual log pattern, based on all tokens
        indices - indices of the initial dataframe, corresponding to current cluster/group of log messages
        cluster_size - number of messages in cluster/group

        The difference between sequence and tokenized_pattern is that tokenized_pattern is used for the
        reconstruction of textual pattern (detokenization), sequence - is a set of cleaned tokens
        and can be used for grouping/clusterization.
        :param gr:
        :return:
        """
        matcher = Match(gr['tokenized_pattern'].values)
        tokenized_pattern = matcher.sequence_matcher(self.add_placeholder)
        return pd.DataFrame([{'indices': gr.index.values.tolist(),
                       'pattern': detokenize_row(tokenized_pattern, self.tokenizer_type),
                       'sequence': gr['sequence'].values[0],
                       'tokenized_pattern': tokenized_pattern,
                       'cluster_size': len(gr.index.values.tolist())}])


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
    def ml_clusterization(self):

        self.clusters = MLClustering(self.df, self.groups,
                                     self.vectors, self.cpu_number, self.add_placeholder,
                                     self.algorithm)
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





