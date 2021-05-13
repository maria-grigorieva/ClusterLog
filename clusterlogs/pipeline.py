import os
import hashlib
import multiprocessing
import numpy as np
import pandas as pd

from time import time

from .reporting import report
from .validation import Output
from .tokenization import tokenize_messages, detokenize_messages, get_term_frequencies, detokenize_row
from .data_preparation import clean_messages
from .sequence_matching import Match
from .ml_clusterization import MLClustering
from .similarity_clusterization import SClustering
from .categorization import execute_categorization
from .phraser import extract_common_phrases
from .utility import gather_df

comm = None
comm_size = 1
comm_rank = 0

if os.environ.get("USE_MPI"):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

default_cleaning_patterns = [
    # Anything other than whitespace that contains '.' inside,
    # not including words that start or end in '.'
    r'\S+\.\S+',
    # Words that contain at least one digit inside but not in the last place
    r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+',
    # Every symbol other than letters, whitespace and '_'
    r'[^\w\s]'
]


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


WORD2VEC_DEFAULTS = {"w2v_size": 300,
                     "w2v_window": 7,
                     "min_samples": 1}


class Chain(object):

    def __init__(self, df, target, *,
                 cleaning_patterns=default_cleaning_patterns,
                 tokenizer_type='space',
                 cluster_settings=None,
                 vectorization_method='word2vec',
                 model_name='word2vec.model',
                 mode='create',
                 output_type='csv',
                 output_fname='report',
                 add_placeholder=True,
                 dimensionality_reduction=False,
                 threshold=5000,
                 matching_accuracy=0.8,
                 clustering_type='dbscan',
                 clustering_parameters={},
                 keywords_extraction='rake_nltk',
                 categorization=False):
        self.df = df
        self.target = target
        self.cleaning_patterns = [] if cleaning_patterns is None else cleaning_patterns
        self.tokenizer_type = tokenizer_type
        self.set_cluster_settings(cluster_settings or WORD2VEC_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.timings = {}
        self.vectorization_method = vectorization_method
        self.model_name = model_name
        self.mode = mode
        self.threshold = threshold
        self.matching_accuracy = matching_accuracy
        self.clustering_type = clustering_type
        self.clustering_parameters = clustering_parameters
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
        for key, value in WORD2VEC_DEFAULTS.items():
            if params.get(key) is not None:
                setattr(self, key, params.get(key))
            else:
                setattr(self, key, value)

    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        """
        # Adds 'tokenized_pattern' column to the dataframe
        # * This can be done after gathering the dataframe, that should be much faster
        self.tokenization()
        # Adds 'hash' and 'sequence' columns
        self.cleaning()
        self.df = gather_df(comm, self.df)
        if comm_rank == 0:
            # Creates a new dataframe - self.groups with
            # 'indices', 'pattern', 'sequence', 'tokenized_pattern', 'cluster_size' columns
            self.group_equals(self.df, 'hash')
            # ? Why do we need the threshold parameter?
            # ? If someone wants to use similarity on big file, it seems logical to let them,
            # ? maybe warn them at most. Especially considering this parameter is user-supplied
            if self.clustering_type == 'similarity' and self.groups.shape[0] <= self.threshold:
                # Creates a self.result dataframe with the following columns:
                # 'pattern', 'tokenized_pattern', 'indices', 'cluster_size', 'sequence', 'common_phrases'
                self.similarity_clustering()
            else:
                # Adds self.vectors which has word2vec and sent2vec attributes
                self.vectorize_messages()
                # Adds 'cluster' column to self.groups
                # and creates self.knee_data if DBSCAN is used
                self.ml_clustering()
                # Creates a self.result dataframe with the following columns:
                # 'patterns', 'pattern_indices', 'indices', 'cluster_size', 'cluster_number', 'common_phrases'
                self.clusters_description()

                self.df["common_pattern"] = None
                self.df["key_phrases"] = None
                self.df["cluster_num"] = None

                self.df["key_phrases"] = self.df["key_phrases"].astype('object')

                # ? Why a function if it is used only once?
                def extend_source(x):
                    for p, indices in x["pattern_indices"].items():
                        self.df.loc[indices, "common_pattern"] = p
                        self.df.loc[indices, "key_phrases"] = str(x["common_phrases"])
                        self.df.loc[indices, "cluster_num"] = x["cluster_number"]

                self.result.apply(func=extend_source, axis="columns")

                self.df.drop(columns=["tokenized_pattern", "hash"], inplace=True)
                # self.df.drop(columns=["tokenized_pattern", "hash", "sequence"], inplace=True)

                self.df.to_csv(f'{self.output_fname}.orig.csv')

            self.process_timings()

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
    def tokenization(self):
        self.df['tokenized_pattern'] = tokenize_messages(self.df[self.target].values, self.tokenizer_type)

    @safe_run
    def cleaning(self):
        if self.cleaning_patterns:
            cleaned_strings = clean_messages(self.df[self.target].values, self.cleaning_patterns)
        else:
            cleaned_strings = self.df[self.target].values
        self.df['cleaned_string'] = cleaned_strings
        # cleaned_tokens = tokenize_messages(cleaned_strings, self.tokenizer_type, spacer_annotate=False, spacer_new=False)
        self.df['hash'] = self.generateHash(cleaned_strings)
        # self.df['sequence'] = cleaned_tokens

    # ! Not used anywhere
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
        sequence = tokenize_messages(self.groups['cleaned_string'], self.tokenizer_type, spacer_annotate=False, spacer_new=False)
        self.groups['sequence'] = sequence
        self.groups.drop(columns=['cleaned_string'], inplace=True)
        print('Found {} equal groups of cleaned messages'.format(self.groups.shape[0]))

    @safe_run
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
        if self.cleaning_patterns:
            matcher = Match(match_threshhold=self.matching_accuracy,
                            add_placeholder=self.add_placeholder)
            # pprint.pprint(gr['tokenized_pattern'].values)
            tokenized_pattern = matcher.sequence_matcher(gr['tokenized_pattern'].values)
            pattern = detokenize_row(tokenized_pattern, self.tokenizer_type)
        else:
            tokenized_pattern = gr['tokenized_pattern'].values[0]
            pattern = gr[self.target].values[0]

        df = pd.DataFrame([{'indices': gr.index.values.tolist(),
                            'pattern': pattern,
                            # 'sequence': gr['sequence'].values[0],
                            'cleaned_string': gr['cleaned_string'].values[0],
                            'tokenized_pattern': tokenized_pattern,
                            'cluster_size': len(gr.index.values.tolist())}])
        return df

    @safe_run
    def vectorize_messages(self):
        if self.vectorization_method == 'word2vec':
            self.w2v_tokens_vectorization()
            self.w2v_sentence_vectorization()
        elif self.vectorization_method == 'bert':
            self.bert_vectorization()
        else:
            print(f"No vectorization method '{self.vectorization_method}' found")

    @safe_run
    def bert_vectorization(self):
        from .vectorization import BertVectorization
        self.vectors = BertVectorization(self.groups['sequence'].values, self.model_name)
        self.vectors.vectorize_messages()

    @safe_run
    def w2v_tokens_vectorization(self):
        """
        Training word2vec model
        :param iterations:
        :param min_count: minimum frequency count of words (recommended value is 1)
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
    def w2v_sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        self.vectors.vectorize_messages()
        print('Vectorization of messages is finished')
        return self

    @safe_run
    def ml_clustering(self):

        self.clusters = MLClustering(
            self.df,
            self.groups,
            self.vectors,
            self.cpu_number,
            self.add_placeholder,
            self.clustering_type,
            self.tokenizer_type,
            self.dimensionality_reduction,
            self.clustering_parameters
        )
        self.clusters.process()

    def clusters_description(self):
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.clusters_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)

    def clusters_regroup(self, gb):
        pattern_indices = self.search_common_patterns(gb)

        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)

        phrases = self.search_keyphrases(pattern_indices.keys())

        return {'patterns': list(pattern_indices.keys()),
                'pattern_indices': pattern_indices,
                'indices': indices,
                'cluster_size': size,
                'cluster_number': gb['cluster'].values[0],
                'common_phrases': phrases[:10]}

    @safe_run
    def search_common_patterns(self, gb):
        m = Match(match_threshhold=self.matching_accuracy,
                  add_placeholder=self.add_placeholder)
        sequences = gb['tokenized_pattern'].values
        indices = gb['indices'].values

        if len(sequences) > 1:
            tokenized_pattern, indices = m.matching_clusters(sequences, indices)
        elif len(sequences) == 1:
            tokenized_pattern, indices = [sequences[0]], [indices[0]]
        else:
            tokenized_pattern, indices = [[]], [[]]

        pattern_indices = dict()

        patterns = detokenize_messages(tokenized_pattern, self.tokenizer_type)

        for p, i in zip(patterns, indices):
            pattern_indices[p] = i

        return pattern_indices

    @safe_run
    def search_keyphrases(self, pattern):
        return extract_common_phrases(pattern, self.keywords_extraction, self.cleaning_patterns)

    # ! Not used anywhere
    def in_cluster(self, groups, cluster_label):
        indices = groups.loc[cluster_label, 'indices']
        return self.df.loc[indices][self.target].values

    @safe_run
    def similarity_clustering(self):
        clusters = SClustering(self.groups,
                               self.matching_accuracy,
                               self.add_placeholder,
                               self.tokenizer_type,
                               self.keywords_extraction)
        self.result = clusters.process()
        print('Finished with {} clusters'.format(self.result.shape[0]))

    # ! This class has several default parameters that do not correspond to Chain
    # ! It is also unused so far
    @safe_run
    def validation(self, groups):
        return Output().statistics(self.df, self.target, groups)

    # ! Not used anywhere
    @staticmethod
    def split_clusters(df, column, threshold=100):
        if np.max(df[column].values) < threshold:
            return df, None
        else:
            return df[df[column] >= threshold], df[df[column] < threshold]

    def process_timings(self):
        if self.timings['group_equals'] != 0 and self.timings['regroup'] != 0:
            self.timings['group_equals'] -= self.timings['regroup']  # group_equals contains regroup

        print(f"Timings:\n{self.timings}")
