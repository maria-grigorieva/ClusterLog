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
            patterns.append({'cluster_name': item,
                            'cluster_size': cluster.shape[0],
                            'pattern': cluster['pattern'].values[0],
                            'sequence': cluster['pattern'].values[0],
                            'mean_similarity': 1,
                            'std_similarity': 0,
                            'indices': cluster.index.values})
            #self.patterns_extraction(item, cluster, patterns)
        return pd.DataFrame(patterns, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)


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


    def patterns_extraction(self, item, cluster, results):
        commons = self.matcher(cluster['tokenized'].values)
        similarity = self.levenshtein_similarity(cluster['cleaned'].values, 0)
        results.append({'cluster_name': item,
                         'cluster_size': cluster.shape[0],
                         'pattern': self.tokenizer.detokenize(commons),
                         'sequence': commons,
                         'mean_similarity': np.mean(similarity),
                         'std_similarity': np.std(similarity),
                         'indices': cluster.index.values})



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



    def split_clusters(self, df):
        if max(df['cluster_size'].values) < 100:
            self.clusters = df
        else:
            self.clusters = df[df['cluster_size'] >= 100]
            self.outliers = df[df['cluster_size'] < 100]
        return self



    def sequence_matcher(self, groups, result):
        sequences = np.array(groups['sequence'].values)
        matched_ids = [idx for idx,score in enumerate(self.levenshtein_similarity(sequences, 10)) if score >= CLUSTERING_ACCURACY]
        result.append([item for i,x in enumerate(groups['indices'].values) for item in x if i in matched_ids])
        #groups = [row for i,row in enumerate(groups) if i not in matched_ids]
        groups.drop(matched_ids, inplace=True)

        if groups.shape[0] > 0:
            self.sequence_matcher(groups, result)







