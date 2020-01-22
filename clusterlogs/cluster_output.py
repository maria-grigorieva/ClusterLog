import editdistance
import difflib
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer


STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "sequence",
              #"mean_length",
              "mean_similarity",
              #"std_length",
              "std_similarity",
              "indices"]

class Output:

    def __init__(self, df, target):
        self.df = df
        self.target = target


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
        clustered_df = self.clustered_output('all')
        for item in clustered_df:
            cluster = clustered_df[item]
            patterns = []
            self.tokenized_patterns(item, cluster, patterns)
            clusters.extend(patterns)
        return pd.DataFrame(clusters, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)



    def result_statistics(self):
        clusters = []
        clustered_df = self.clustered_output('all')
        for item in clustered_df:
            cluster = clustered_df[item]
            similarity = self.get_cluster_similarity(cluster['pattern'].values)
            indices = [item for sublist in cluster['indices'].values for item in sublist]
            clusters.append({'cluster_name': item,
                             'cluster_size': len(indices),
                             'pattern': cluster['pattern'].values[0],
                             'sequence': cluster['tokenized_pyonmttok'].values[0],
                             'mean_similarity': np.mean(similarity),
                             'std_similarity': np.std(similarity),
                             'indices': indices})
        return pd.DataFrame(clusters, columns=STATISTICS) \
            .round(2) \
            .sort_values(by='cluster_size', ascending=False)



    def get_cluster_similarity(self, patterns):
        return [1] if len(patterns) == 1 else \
            [difflib.SequenceMatcher(None, patterns[0], patterns[i]).ratio() for i in range(1,len(patterns))]


    def positioning(self, tokens):
        return [(v, k) for k, v in enumerate(tokens)]


    def tokenized_patterns(self, item, cluster, results):

        if cluster.shape[0] == 1:
            value = {'cluster_name': item,
                     'cluster_size': 1,
                     'pattern': cluster.iloc[0]['cleaned'],
                     'sequence': cluster.iloc[0]['tokenized_pyonmttok'],
                     'mean_similarity': 1,
                     'std_similarity': 0,
                     'indices': cluster.index.values}
            results.append(value)
            return results
        else:
            curr = cluster.iloc[0]['tokenized_pyonmttok']
            outliers, commons, similarity = [],[],[]
            [self.matcher(curr, row, commons, similarity) for row in cluster.itertuples()]
            commons = self.rematch(commons)
            twd = TreebankWordDetokenizer()

            value = {'cluster_name': item,
                     'cluster_size': cluster.shape[0] - len(outliers),
                     'pattern': twd.detokenize(commons),
                     'sequence': commons,
                     'mean_similarity': np.mean(similarity),
                     'std_similarity': np.std(similarity),
                     'indices': cluster.index.values}

            results.append(value)


    #TODO
    def rematch(self, commons):
        """
        First we found matches in strings.
        Next we find matches in matches.
        :param commons:
        :return:
        """
        arr = {}
        sorted_arr = [sorted(set([val for sublist in commons for val in sublist]), key=lambda x: x[1])]
        for i in sorted_arr:
            for k,v in i:
                if arr.get(v):
                    arr[v].append(k)
                else:
                    arr[v] = []
                    arr[v].append(k)

        return [arr[s][0] for s in arr]


    def matcher(self, current, row, commons, similarity):

        matches = difflib.SequenceMatcher(None, current, row.tokenized_pyonmttok)
        similarity.append(matches.ratio())
        common = [current[m.a:m.a + m.size] for m
                  in matches.get_matching_blocks() if m.size > 0]
        flat = [val for sublist in common for val in sublist]
        commons.append(self.positioning(flat))


    def reclustering(self, df, result, counter=0):
        """
        :param df:
        :param updated_clusters:
        :return:
        """
        sequences = df['sequence'].values
        matches = [difflib.SequenceMatcher(None, sequences[0], x) for x in sequences]
        df['ratio'] = [item.ratio() for item in matches]
        filtered = df[(df['ratio'] >= 0.5)]
        # indices = [item for sublist in filtered['indices'].values for item in sublist] if filtered.shape[0]>0 else
        new_cluster = {}
        new_cluster['cluster_name'] = counter
        new_cluster['indices'] = [item for sublist in filtered['indices'].values for item in sublist]
        result.append(new_cluster)
        df.drop(df[df['cluster_name'].isin(filtered['cluster_name'].values)].index, inplace=True)
        counter += 1
        while df.shape[0] > 0:
            self.reclustering(df, result, counter)


    def get_common_pattern(self, sequences):
        curr = sequences[0]

        for i in range(1, len(sequences)):
            matches = difflib.SequenceMatcher(None, curr, sequences[i])
            common = [curr[m.a:m.a + m.size] for m
                      in matches.get_matching_blocks() if m.size > 0]
            curr = [val for sublist in common for val in sublist]

        return curr


    def postprocessing(self, stat_df):
        """
        Clustering the results of the first clusterization
        :return:
        """
        # sort statistics df by cluster size in ascending order
        sorted_df = stat_df.sort_values(by=['cluster_size'])[['cluster_size',
                                                              'cluster_name',
                                                              'pattern',
                                                              'sequence',
                                                              'indices']]
        result = []
        print("total number of clusters is " + str(stat_df.shape[0]))
        self.reclustering(sorted_df, result)
        print("new number of clusters is " + str(len(result)))

        total = 0
        for item in result:
            self.df.loc[item['indices'], 'cluster'] = item['cluster_name']
            total += len(item['indices'])
        print('total = ' + str(total))
        print('df size = ' +str(self.df.shape[0]))

        print("result in df is " + str(len(self.df.groupby('cluster'))))

        return self.statistics()










