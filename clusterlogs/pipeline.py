import pprint
import hashlib
import multiprocessing
import numpy as np
import pandas as pd

from time import time
from string import punctuation

from .reporting import report
from .validation import Output
from .tokenization import tokenize_messages, get_term_frequencies, detokenize_row
from .data_preparation import clean_messages
from .sequence_matching import Match
from .ml_clusterization import MLClustering
from .similarity_clusterization import SClustering
from .categorization import execute_categorization


def safe_run(method):

    def func_wrapper(self, *args, **kwargs):
        try:
            ts = time()
            result = method(self, *args, **kwargs)
            te = time()
            if method.__name__ in self.timings:
                self.timings[method.__name__] += round((te - ts), 4)
            else:
                self.timings[method.__name__] = round((te - ts), 4)
            return result
        except Exception as e:
            print(f"{method.__name__} threw an exception: {e}")

    return func_wrapper


CLUSTERING_DEFAULTS = {"w2v_size": 300,
                       "w2v_window": 7,
                       "min_samples": 1}


class Chain(object):

    CLUSTERING_THRESHOLD = 5000
    MATCHING_ACCURACY = 0.8
    CLUSTERING_TYPE = 'dbscan'
    # 'similarity', 'hdbscan', 'hierarchical'
    KEYWORDS_EXTRACTION = 'rake_nltk'

    def __init__(self, df, target,
                 tokenizer_type='space',
                 cluster_settings=None,
                 model_name='word2vec.model',
                 mode='create',
                 output_type='csv',
                 output_fname='report',
                 add_placeholder=True,
                 dimensionality_reduction=False,
                 threshold=CLUSTERING_THRESHOLD,
                 matching_accuracy=MATCHING_ACCURACY,
                 clustering_type=CLUSTERING_TYPE,
                 keywords_extraction=KEYWORDS_EXTRACTION,
                 categorization=False):
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
        self.output_type = output_type
        self.output_fname = output_fname
        self.categorization = categorization
        self.dimensionality_reduction = dimensionality_reduction
        self.keywords_extraction = keywords_extraction

    @staticmethod
    def get_cpu_number():
        return multiprocessing.cpu_count()

    def set_cluster_settings(self, params):
        for key, value in CLUSTERING_DEFAULTS.items():
            if params.get(key) is not None:
                setattr(self, key, params.get(key))
            else:
                setattr(self, key, value)

    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        """
        self.df['tokenized_pattern'] = tokenize_messages(self.df[self.target].values, self.tokenizer_type)
        cleaned_strings = clean_messages(self.df[self.target].values)
        cleaned_tokens = tokenize_messages(cleaned_strings, self.tokenizer_type, spacer_annotate=False, spacer_new=False)

        self.df['hash'] = self.generateHash(cleaned_strings)
        self.df['sequence'] = cleaned_tokens

        self.group_equals(self.df, 'hash')

        if self.clustering_type == 'similarity' and self.groups.shape[0] <= self.threshold:
            self.similarity_clustering()
        else:
            self.tokens_vectorization()
            self.sentence_vectorization()
            self.ml_clustering()

        print(f"Timings:\n{self.timings}")

        # Categorization
        fname = f'{self.output_fname}.{self.output_type}'
        if self.output_type == 'html':
            if self.categorization:
                self.categories = execute_categorization(self.result)
                report.categorized_report(self.categories, fname)
            else:
                report.generate_html_report(self.result, fname)
        elif self.output_type == 'csv':
            self.result.to_csv(fname)

    @safe_run
    def remove_unique_tokens(self, tokens):
        frequency = get_term_frequencies(tokens)
        # remove tokens that appear only once
        cleaned_tokens = [
            [token for token in row if frequency[token] > 1]
            for row in tokens]
        return cleaned_tokens

    def generateHash(self, sequences):
        return [hashlib.md5(repr(row).encode('utf-8')).hexdigest() for row in sequences]

    @safe_run
    def group_equals(self, df, column):
        self.groups = df.groupby(column).apply(func=self.regroup)
        self.groups.reset_index(drop=True, inplace=True)
        print('Found {} equal groups'.format(self.groups.shape[0]))

    def regroup(self, gr):
        """
        tokenized_pattern - common sequence of tokens, generated based on all tokens
        sequence - common sequence of tokens, based on cleaned tokens
        pattern - textual log pattern, based on all tokens
        indices - indices of the initial dataframe, corresponding to current cluster/group of log messages
        cluster_size - number of messages in cluster/group

        The difference between sequence and tokenized_pattern is that tokenized_pattern is used for the
        reconstruction of textual pattern (detokenization), while sequence is a set of cleaned tokens
        and can be used for grouping/clusterization.
        :param gr:
        :return:
        """
        matcher = Match(gr['tokenized_pattern'].values, add_placeholder=self.add_placeholder)
        # pprint.pprint(gr['tokenized_pattern'].values)
        tokenized_pattern = matcher.sequence_matcher()
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
            self.vectors.create_word2vec_model()
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
        self.vectors.vectorize_messages(tf_idf=False)
        print('Vectorization of sentences is finished')
        return self

    @safe_run
    def ml_clustering(self):

        self.clusters = MLClustering(self.df,
                                     self.groups,
                                     self.vectors,
                                     self.cpu_number,
                                     self.add_placeholder,
                                     self.clustering_type,
                                     self.tokenizer_type,
                                     self.dimensionality_reduction,
                                     self.keywords_extraction)
        self.result = self.clusters.process()

    def in_cluster(self, groups, cluster_label):
        indices = groups.loc[cluster_label, 'indices']
        return self.df.loc[indices][self.target].values

    @safe_run
    def similarity_clustering(self):
        clusters = SClustering(self.groups,
                               self.matching_accuracy,
                               self.add_placeholder,
                               self.tokenizer_type)
        self.result = clusters.process()
        print('Finished with {} clusters'.format(self.result.shape[0]))

    @safe_run
    def validation(self, groups):
        return Output().statistics(self.df, self.target, groups)

    @staticmethod
    def split_clusters(df, column, threshold=100):
        if np.max(df[column].values) < threshold:
            return df, None
        else:
            return df[df[column] >= threshold], df[df[column] < threshold]
