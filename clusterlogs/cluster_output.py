import editdistance
import difflib
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random


STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "sequence",
              "mean_similarity",
              "std_similarity",
              "indices"]

class Output:

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.patterns = None


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
        return ([100 - (editdistance.eval(rows[0], rows[i]) * 100) / len(rows[0]) for i in range(0, len(rows))])


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
            self.tokenized_patterns(item, cluster, patterns)
        self.patterns = pd.DataFrame(patterns, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)
        return self.patterns


    def tokenized_patterns(self, item, cluster, results):

        if cluster.shape[0] == 1:
            value = {'cluster_name': item,
                     'cluster_size': 1,
                     'pattern': cluster.iloc[0][self.target],
                     'sequence': cluster.iloc[0]['tokenized_pyonmttok'],
                     'mean_similarity': 1,
                     'std_similarity': 0,
                     'indices': cluster.index.values}
            results.append(value)
        else:
            commons, similarity = self.matcher(cluster)
            twd = TreebankWordDetokenizer()

            value = {'cluster_name': item,
                     'cluster_size': cluster.shape[0],
                     'pattern': twd.detokenize(commons),
                     'sequence': commons,
                     'mean_similarity': np.mean(similarity),
                     'std_similarity': np.std(similarity),
                     'indices': cluster.index.values}

            results.append(value)


    def matcher(self, cluster):
        lines = cluster['tokenized_pyonmttok'].values
        x = random.choice(lines)
        y = random.choice(lines)
        matches = difflib.SequenceMatcher(None, x, y)
        similarity = matches.ratio()
        common = [x[m.a:m.a + m.size] for m
                  in matches.get_matching_blocks() if m.size > 0]
        return [val for sublist in common for val in sublist], similarity
        #
        #
        # similarity = []
        # current = cluster.iloc[0]['tokenized_pyonmttok']
        # for i in range(1, cluster.shape[0]):
        #     matches = difflib.SequenceMatcher(None, current, cluster.iloc[i]['tokenized_pyonmttok'])
        #     common = [current[m.a:m.a + m.size] for m
        #               in matches.get_matching_blocks() if m.size > 0]
        #     similarity.append(matches.ratio())
        #     current = [val for sublist in common for val in sublist]
        # return current, similarity



    def reclustering(self, df, result):
        curr = df['sequence'].values[0]
        matches = [difflib.SequenceMatcher(None, curr, x) for x in df['sequence'].values]
        df['ratio'] = [item.ratio() for item in matches]
        filtered = df[(df['ratio'] >= 0.8)]
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        result.append(indices)
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)


    def postprocessing(self, clusters):

        result = []
        self.reclustering(clusters.copy(deep=True), result)

        for i in range(0, len(result)):
            self.df.loc[result[i], 'cluster'] = i

        self.statistics()









