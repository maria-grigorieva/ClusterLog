from time import time
from pyonmttok import Tokenizer
import editdistance
import nltk
import numpy as np
import pandas as pd
from itertools import groupby
import re


CLUSTERING_ACCURACY = 0.8

STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
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


class ml_clustering(object):


    def __init__(self, df, target):
        self.df = df
        self.df['cluster'] = 0
        self.target = target
        self.messages = df[self.target].values
        self.timings = {}
        self.messages_cleaned = None
        self.tokens = None
        self.results = None
        self.tokenizer = Tokenizer("conservative", spacer_annotate=True)


    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        :return:
        """
        return self.data_preparation() \
            .tokenization() \
            .group_equals() \
            .grouped_sequences() \
            .split_clusters()



    @safe_run
    def data_preparation(self):
        """
        Cleaning log messages from unnucessary substrings and tokenization
        :return:
        """
        self.messages_cleaned = [0] * len(self.messages)
        for idx, item in enumerate(self.messages):
            try:
                item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '*', item)
                self.messages_cleaned[idx] = item
            except Exception as e:
                print(item)
        self.df['cleaned'] = self.messages_cleaned
        return self


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        :return:
        """
        self.df['tokenized'] = self.pyonmttok(self.messages)
        self.df['tokenized_cleaned'] = self.pyonmttok(self.messages_cleaned)
        self.vocabulary = self.get_vocabulary(self.df['tokenized'].values)
        self.vocabulary_cleaned = self.get_vocabulary(self.df['tokenized_cleaned'].values)

        return self


    def in_cluster(self, cluster_label):
        return self.df[self.df['cluster'] == cluster_label][self.target].values


    @safe_run
    def group_equals(self):

        gp = self.df.groupby('cleaned')
        self.messages = [i for i in gp.groups]
        groups = []
        for key, value in gp:
            indices = value.index.values.tolist()
            pattern = value['cleaned'].values[0]
            sequence = value['tokenized_cleaned'].values[0]
            groups.append({'pattern':pattern, 'sequence': sequence, 'indices':indices})
        self.groups = pd.DataFrame(groups)

        return self


    def grouped_sequences(self):

        result = []
        self.sequence_matcher(self.groups, result)
        for i,row in enumerate(result):
            self.df.loc[row, 'cluster'] = i

        self.results = self.statistics()

        return self


    @safe_run
    def sequence_matcher(self, groups, result):
        sequences = np.array(groups['sequence'].values)
        matched_ids = [idx for idx,score in enumerate(self.levenshtein_similarity(sequences, 0)) if score >= CLUSTERING_ACCURACY]
        result.append([item for i,x in enumerate(groups['indices'].values) for item in x if i in matched_ids])
        groups.drop(matched_ids, inplace=True)
        groups.reset_index(inplace=True, drop=True)

        if groups.shape[0] > 0:
            self.sequence_matcher(groups, result)


    def split_clusters(self):
        if max(self.results['cluster_size'].values) < 100:
            self.clusters = self.results
        else:
            self.clusters = self.results[self.results['cluster_size'] >= 100]
            self.outliers = self.results[self.results['cluster_size'] < 100]
        return self



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


    @safe_run
    def statistics(self):
        """
        :param clustered_df:
        :param output_mode: data frame
        :return:
        """
        patterns = []
        clustered_df = self.clustered_output('all','cluster')
        for item in clustered_df:
            cluster = clustered_df[item]
            self.patterns_extraction(item, cluster, patterns)
        return pd.DataFrame(patterns, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)


    def patterns_extraction(self, item, cluster, results):
        commons = self.matcher(cluster['tokenized'].values)
        similarity = self.levenshtein_similarity(cluster['cleaned'].values, 0)
        results.append({'cluster_name': item,
                         'cluster_size': cluster.shape[0],
                         'pattern': self.tokenizer.detokenize(commons),
                         'mean_similarity': np.mean(similarity),
                         'std_similarity': np.std(similarity),
                         'indices': cluster.index.values})


    def matcher(self, lines):
        if len(lines) > 1:
            fdist = nltk.FreqDist([i for l in lines for i in l])
            x = [token if (fdist[token]/len(lines) >= 1) else '{*}' for token in lines[0]]
            return [i[0] for i in groupby(x)]
        else:
            return lines[0]


    def pyonmttok(self, strings):
        tokenized = []
        for line in strings:
            tokens, features = self.tokenizer.tokenize(line)
            tokenized.append(tokens)
        return tokenized


    def tokenize_string(self, line):
        tokens, features = self.tokenizer.tokenize(line)
        return tokens


    def get_vocabulary(self, tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))


    def clustered_output(self, type='idx', column='cluster'):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        if type != 'idx':
            groups = {}
            for key, value in self.df.groupby([column]):
                if type == 'all':
                    groups[str(key)] = value
                elif type == 'target':
                    groups[str(key)] = value[self.target].values.tolist()
                elif type == 'cleaned':
                    groups[str(key)] = value['cleaned'].values.tolist()
            return groups
        else:
            return [[value['tokenized_cleaned'].values.tolist()[0], value.index.values.tolist()] for key, value in self.df.groupby([column])]