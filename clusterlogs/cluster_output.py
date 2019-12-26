import editdistance
import difflib
import numpy as np
import pandas as pd
from .tokenization import Tokens
import pprint

STATISTICS = ["cluster_name", "cluster_size", "pattern",
              "mean_length", "mean_similarity", "std_length", "std_similarity"]

class Output:

    def __init__(self, df, target, level=1):
        self.df = df
        self.target = target
        self.level = level
        self.stat_df = None
        self.stat_dict = None
        self.hierarchy = None


    def clustered_output(self, mode='INDEX', level=1):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        groups = {}
        for key, value in self.df.groupby(['cluster_'+str(level)]):
            if mode == 'all':
                groups[str(key)] = value.to_dict(orient='records')
            elif mode == 'idx':
                groups[str(key)] = value.index.values.tolist()
            elif mode == 'target':
                groups[str(key)] = value[self.target].values.tolist()
            elif mode == 'cleaned':
                groups[str(key)] = value['_cleaned'].values.tolist()
        return groups


    def in_cluster(self, cluster_label, level=1):
        """
        Returns all log messages in particular cluster
        :param cluster_label:
        :return:
        """
        return self.df[self.df['cluster_'+str(level)] == cluster_label][self.target].values


    def levenshtein_similarity(self, rows):
        """
        Takes a list of log messages and calculates similarity between
        first and all other messages.
        :param rows:
        :return:
        """
        return ([100 - (editdistance.eval(rows[0], rows[i]) * 100) / len(rows[0]) for i in range(0, len(rows))])


    def statistics(self, output_mode='frame', level=1):
        """
        Returns dictionary with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "pattern" - all common substrings in messages in the cluster
        "mean_length" - average length of log messages in cluster
        "std_length" - standard deviation of length of log messages in cluster
        "mean_similarity" - average similarity of log messages in cluster
        "std_similarity" - standard deviation of similarity of log messages in cluster
        "interbals" - internal indexes of the cluster
        :param clustered_df:
        :param output_mode: frame | dict
        :return:
        """
        clusters = []
        clustered_df = self.clustered_output('cleaned', level)
        for item in clustered_df:
            row = clustered_df[item]
            matcher, similarity = self.matcher(row)
            lengths = [len(s) for s in row]
            clusters.append([item,
                             len(row),
                             matcher,
                             np.mean(lengths),
                             np.mean(similarity),
                             np.std(lengths) if len(row) > 1 else 0,
                             np.std(similarity)])
        self.stat_df = pd.DataFrame(clusters, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)
        self.stat_dict = self.stat_df.to_dict(orient='records')
        if output_mode == 'frame':
            return self.stat_df
        else:
            return self.stat_dict


    def matcher(self, strings):
        """
        Find all matching blocks in a list of strings
        :param strings:
        :return:
        """
        similarity = []
        curr = strings[0]
        if len(strings) == 1:
            return curr, 1
        else:
            cnt = 1
            for i in range(cnt, len(strings)):
                matches = difflib.SequenceMatcher(None, curr, strings[i])
                similarity.append(matches.ratio())
                common = []
                for match in matches.get_matching_blocks():
                    common.append(curr[match.a:match.a + match.size])
                curr = ''.join(common)
                cnt = cnt + 1
                if cnt == len(strings):
                    break
            return curr, similarity


    @staticmethod
    def similarity(rows):
        """
        Return a measure of the sequences’ similarity as a float in the range [0, 100] percent
        :param strings:
        :return:
        """
        s = []
        for i in range(0, len(rows)):
            s.append(difflib.SequenceMatcher(None, rows[0], rows[i]).ratio() * 100)
        return s


    def first_match(self, df, result):
        """

        :param df:
        :param updated_clusters:
        :return:
        """
        # select the first pattern
        start = df.head(1)['pattern'].values[0]
        matches = [difflib.SequenceMatcher(None, start, x) for x in df['pattern']]
        ratio = []
        is_first = []
        for item in matches:
            ratio.append(item.ratio())
            is_first.append(item.get_matching_blocks()[0].a == 0)
        df['ratio'] = ratio
        df['is_first'] = is_first
        filtered = df[(df['ratio'] > 0.5) & (df['is_first']==True)]['cluster_name'].values
        result.append(filtered)
        df.drop(df[df['cluster_name'].isin(filtered)].index, inplace=True)
        while df.shape[0] > 0:
            self.first_match(df, result)


    def postprocessing(self, level):
        """
        Clustering the results of the first clusterization
        :return:
        """
        # sort statistics df by cluster size in ascending order
        sorted_df = self.stat_df.sort_values(by=['cluster_size'])[['cluster_size',
                                                                   'cluster_name',
                                                                   'pattern']]
        result = []
        self.first_match(sorted_df, result)
        new_level = []
        for k,v in enumerate(result):
            x = self.df[self.df['cluster_'+str(level)].isin(v)].index
            new_level.append({'cluster_name': k, 'idx': x})
        for cluster in new_level:
            self.df.loc[cluster['idx'], 'cluster_'+str(level+1)] = cluster['cluster_name']

        return self.statistics(output_mode='frame', level=level+1)

