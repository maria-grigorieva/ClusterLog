comm = None
comm_size = 1
comm_rank = 0

import os
if os.environ.get("USE_MPI"):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

parallel_profile = dict()

from time import time

def time_start(key):
    parallel_profile[key] = time()    

def time_stop(key):
    parallel_profile[key] = time() - parallel_profile[key]

time_start("pipeline_def_time")
time_start("pipeline_import_time")

import hashlib
import multiprocessing
import numpy as np
import pandas as pd

import math

# from string import punctuation

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

time_stop("pipeline_import_time")

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

    def __init__(self, df, target,
                 tokenizer_type='space',
                 cluster_settings=None,
                 model_name='word2vec.model',
                 mode='create',
                 output_type='csv',
                 output_fname='report',
                 add_placeholder=True,
                 dimensionality_reduction=False,
                 threshold=5000,
                 matching_accuracy=0.8,
                 clustering_type='dbscan',
                 keywords_extraction='rake_nltk',
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
        
        time_start("messages_index_time")
        # e2e indexing for df
        self.df.set_index(pd.RangeIndex(comm_rank, comm_rank + comm_size * self.df.shape[0], comm_size), inplace=True)
        time_stop("messages_index_time")

        parallel_profile["justloaded_messages_size"] = int(self.df.memory_usage(deep=True, index=True).sum())
        parallel_profile["justloaded_messages_num"] = self.df.shape[0]

        time_start("tokenizing_time")
        self.tokenization()
        time_stop("tokenizing_time")

        time_start("cleaning_time")
        self.cleaning()
        time_stop("cleaning_time")

        time_start("grouping_time")
        self.group_equals(self.df, 'hash')
        time_stop("grouping_time")
        
        time_start("group_index_time")
        # e2e indexing for groups
        self.groups.set_index(pd.RangeIndex(comm_rank, comm_rank + comm_size * self.groups.shape[0], comm_size), inplace=True)
        time_stop("group_index_time")
    
        time_start("mem_before_gather_time")
        parallel_profile["preprocessed_messages_size"] = int(self.df.memory_usage(deep=True, index=True).sum())
        parallel_profile["preprocessed_messages_num"] = self.df.shape[0]
        parallel_profile["local_groups_size"] = int(self.groups.memory_usage(deep=True, index=True).sum())
        parallel_profile["local_groups_num"] = self.groups.shape[0]
        time_stop("mem_before_gather_time")
        
        time_start("gather_df_time")
        self.df = gather_df(comm, self.df)
        time_stop("gather_df_time")

        time_start("gather_groups_time")
        self.groups = gather_df(comm, self.groups)
        time_stop("gather_groups_time")
        
        if comm_rank == 0:
            parallel_profile["gathered_groups_size"] = int(self.groups.memory_usage(deep=True, index=True).sum())
            parallel_profile["gathered_groups_shape"] = self.groups.shape[0]
            if comm_size > 1:
                time_start("merge_groups_time")
                self.groups["sequence_str"] = self.groups["sequence"].apply(str)
                self.groups = self.groups.groupby("sequence_str").aggregate({"indices": "sum", "pattern": "first",
                                                                             "sequence": "first",
                                                                             "tokenized_pattern": "first",
                                                                             "cluster_size": "sum"}).reset_index()
                time_stop("merge_groups_time")
            else:
                parallel_profile["merge_groups_time"] = 0

            
            time_start("mem_after_gather_time")
            parallel_profile["gathered_messages_size"] = int(self.df.memory_usage(deep=True, index=True).sum())
            parallel_profile["gathered_messages_num"] = self.df.shape[0]
            parallel_profile["merged_groups_size"] = int(self.groups.memory_usage(deep=True, index=True).sum())
            parallel_profile["merged_groups_shape"] = self.groups.shape[0]
            time_stop("mem_after_gather_time")

            time_start("clustering_time")
            if self.clustering_type == 'similarity' and self.groups.shape[0] <= self.threshold:
                self.similarity_clustering()
            else:
                self.tokens_vectorization()
                self.sentence_vectorization()
                self.ml_clustering()
                self.clusters_description()
            time_stop("clustering_time")

            self.process_timings()
            
            parallel_profile["clusters_num"] = self.result.shape[0]

            time_start("outputing_time")
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
            time_stop("outputing_time")
        
        return parallel_profile

    @safe_run
    def tokenization(self):
        self.df['tokenized_pattern'] = tokenize_messages(self.df[self.target].values, self.tokenizer_type)

    @safe_run
    def cleaning(self):
        cleaned_strings = clean_messages(self.df[self.target].values)
        cleaned_tokens = tokenize_messages(cleaned_strings, self.tokenizer_type, spacer_annotate=False, spacer_new=False)
        self.df['hash'] = self.generateHash(cleaned_strings)
        self.df['sequence'] = cleaned_tokens

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
        matcher = Match(match_threshhold=self.matching_accuracy,
                        add_placeholder=self.add_placeholder)
        # pprint.pprint(gr['tokenized_pattern'].values)
        tokenized_pattern = matcher.sequence_matcher(gr['tokenized_pattern'].values)

        df = pd.DataFrame([{'indices': gr.index.values.tolist(),
                              'pattern': detokenize_row(tokenized_pattern, self.tokenizer_type),
                              'sequence': gr['sequence'].values[0],
                              'tokenized_pattern': tokenized_pattern,
                              'cluster_size': len(gr.index.values.tolist())}])
        return df

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
        print('Vectorization of messages is finished')
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
                                     self.dimensionality_reduction)
        self.clusters.process()

    def clusters_description(self):
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.clusters_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)

    def clusters_regroup(self, gb):
        pattern = self.search_common_patterns(gb)

        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)

        phrases = self.search_keyphrases(pattern)

        return {'pattern': pattern,
                'indices': indices,
                'cluster_size': size,
                'common_phrases': phrases[:10]}

    @safe_run
    def search_common_patterns(self, gb):
        m = Match(match_threshhold=self.matching_accuracy,
                  add_placeholder=self.add_placeholder)
        tokenized_pattern = []
        sequences = gb['tokenized_pattern'].values

        if len(sequences) > 1:
            m.matching_clusters(sequences, tokenized_pattern)
        elif len(sequences) == 1:
            tokenized_pattern.append(sequences[0])
        return detokenize_messages(tokenized_pattern, self.tokenizer_type)

    @safe_run
    def search_keyphrases(self, pattern):
        return extract_common_phrases(pattern, self.keywords_extraction)

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

    @safe_run
    def validation(self, groups):
        return Output().statistics(self.df, self.target, groups)

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

time_stop("pipeline_def_time")
