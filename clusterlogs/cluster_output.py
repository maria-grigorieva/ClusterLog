import editdistance
import difflib
import numpy as np
import pandas as pd
from pyonmttok import Tokenizer
import nltk
from itertools import groupby


STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "sequence",
              "mean_similarity",
              "std_similarity",
              "indices"]


CLUSTERING_ACCURACY = 0.8

class Output:

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.patterns = None
        self.tokenizer = Tokenizer("conservative", spacer_annotate=True)


    def clustered_output(self, type='idx'):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        groups = {}
        for key, value in self.df.groupby(['cluster']):
            if type == 'all':
                groups[str(key)] = value
            elif type == 'idx':
                groups[str(key)] = value.index.values.tolist()
            elif type == 'target':
                groups[str(key)] = value[self.target].values.tolist()
            elif type == 'cleaned':
                groups[str(key)] = value['cleaned'].values.tolist()
        return groups


    def levenshtein_similarity(self, rows):
        """
        Takes a list of log messages and calculates similarity between
        first and all other messages.
        :param rows:
        :return:
        """
        if len(rows) > 1:
            return ([100 - (editdistance.eval(rows[0], rows[i]) * 100) / len(rows[0]) for i in range(0, len(rows))])
        else:
            return 100


    def statistics(self):
        """
        Returns dictionary with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "pattern" - all common substrings in messages in the cluster
        "mean_similarity" - average similarity of log messages in cluster
        "std_similarity" - standard deviation of similarity of log messages in cluster
        "indices" - internal indexes of the cluster
        :param clustered_df:
        :param output_mode: data frame
        :return:
        """
        patterns = []
        clustered_df = self.clustered_output('all')
        for item in clustered_df:
            cluster = clustered_df[item]
            self.patterns_extraction(item, cluster, patterns)
        self.patterns = pd.DataFrame(patterns, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)
        return self.patterns


    def patterns_extraction(self, item, cluster, results):
        commons = self.matcher(cluster['tokenized'].values)
        similarity = self.levenshtein_similarity(cluster['cleaned'].values)
        results.append({'cluster_name': item,
                         'cluster_size': cluster.shape[0],
                         'pattern': self.tokenizer.detokenize(commons),
                         'sequence': commons,
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


    def reclustering(self, df, result):

        curr = df['sequence'].values[0]
        df['ratio'] = [difflib.SequenceMatcher(None, curr, x).ratio() for x in df['sequence'].values]
        filtered = df[(df['ratio'] >= CLUSTERING_ACCURACY)]
        result.append([item for sublist in filtered['indices'].values for item in sublist])
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)


    def postprocessing(self, clusters):

        result = []
        self.reclustering(clusters.copy(deep=True), result)

        for i in range(0, len(result)):
            self.df.loc[result[i], 'cluster'] = i

        self.statistics()









