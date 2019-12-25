import editdistance
import difflib
import numpy as np
import pandas as pd
from .tokenization import Tokens

STATISTICS = ["cluster_name", "cluster_size", "pattern",
              "mean_length", "mean_similarity", "std_length", "std_similarity"]

class Output:

    def __init__(self, df, target, tokenizer, cluster_labels):
        self.cluster_labels = cluster_labels
        self.df = df
        self.target = target
        # self.messages = messages
        self.tokenizer = tokenizer
        # self.df[target] = self.messages
        self.df['cluster'] = self.cluster_labels


    def clustered_output(self, mode='INDEX'):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        groups = {}
        # self.df['cluster'] = self.cluster_labels
        for key, value in self.df.groupby(['cluster']):
            if mode == 'ALL':
                groups[str(key)] = value.to_dict(orient='records')
            elif mode == 'INDEX':
                groups[str(key)] = value.index.values.tolist()
            elif mode == 'TARGET':
                groups[str(key)] = value[self.target].values.tolist()
            elif mode == 'CLEANED':
                groups[str(key)] = value['_cleaned'].values.tolist()
        return groups


    def in_cluster(self, cluster_label):
        """
        Returns all log messages in particular cluster
        :param cluster_label:
        :return:
        """
        results = []
        for idx, l in enumerate(self.cluster_labels):
            if l == cluster_label:
                results.append(self.df[self.target].values[idx])
        return results


    def levenshtein_similarity(self, rows):
        """
        Takes a list of log messages and calculates similarity between
        first and all other messages.
        :param rows:
        :return:
        """
        return ([100 - (editdistance.eval(rows[0], rows[i]) * 100) / len(rows[0]) for i in range(0, len(rows))])


    def statistics(self, output_mode='frame'):
        """
        Returns dictionary with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "pattern" - all common substrings in messages in the cluster
        "vocab" - vocabulary of all messages within the cluster (without punctuation and stop words)
        "vocab_length" - the length of vocabulary
        "mean_length" - average length of log messages in cluster
        "std_length" - standard deviation of length of log messages in cluster
        "mean_similarity" - average similarity of log messages in cluster
        (calculated as the levenshtein distances between the 1st and all other log messages)
        "std_similarity" - standard deviation of similarity of log messages in cluster
        :param clustered_df:
        :param output_mode: frame | dict
        :return:
        """
        clusters = []
        clustered_df = self.clustered_output(mode='CLEANED')
        for item in clustered_df:
            row = clustered_df[item]
            matcher, similarity = self.matcher(row)
            lengths = [len(s) for s in row]
            #similarity = self.similarity(row)
            tokens = Tokens(row, self.tokenizer)
            tokens.process()
            # vocab = tokens.get_vocabulary()
            # vocab_length = len(vocab)
            clusters.append([item,
                             len(row),
                             matcher,
                             # vocab,
                             # vocab_length,
                             np.mean(lengths),
                             np.mean(similarity),
                             np.std(lengths) if len(row) > 1 else 0,
                             np.std(similarity)])
        df = pd.DataFrame(clusters, columns=STATISTICS).round(2).sort_values(by='cluster_size', ascending=False)
        if output_mode == 'frame':
            return df
        else:
            return df.to_dict(orient='records')


    @staticmethod
    def matcher(strings):
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