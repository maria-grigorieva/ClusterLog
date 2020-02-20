import pandas as pd
import difflib
import editdistance
from .tokenization import Tokens


CLUSTERING_ACCURACY = 0.8

class SClustering:

    def __init__(self, groups, tokens):
        self.groups = groups
        self.tokens = tokens



    def matching_clusterization(self, accuracy=CLUSTERING_ACCURACY):
        """
        Clusterization messages using sequence matching
        :param df:
        :param accuracy:
        :return:
        """
        result = []
        if self.groups.shape[0] <= 100:
            self.result = self.groups
        else:
            self.reclustering(self.groups.copy(deep=True), result, accuracy)
            self.result = pd.DataFrame(result)
        print('Postprocessed with {} clusters'.format(self.result.shape[0]))
        return self.result.sort_values(by=['cluster_size'], ascending=False)



    def reclustering(self, df, result, accuracy):
        """
        Clusterization of the groups:
        - take the 1st message (pattern) and compare if with others
        - take all messages, which are similar with the 1st with more than 80% and
        join them into the new separate cluster
        - remove these messages from the initial group
        - repeat these steps while group has messages
        :param df:
        :param result:
        :param accuracy:
        :return:
        """
        df['ratio'] = self.levenshtein_similarity(df['tokenized_pattern'].values)
        filtered = df[(df['ratio'] >= accuracy)]
        pattern = self.sequence_matcher(filtered['tokenized_pattern'].values)
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        result.append({'pattern': pattern,
                       'indices': indices,
                       'cluster_size': len(indices)})
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result, accuracy)



    def gb_regroup(self, gb):
        common_pattern = self.sequence_matcher(gb['tokenized_pattern'].values)
        sequence = self.tokens.tokenize_string(self.tokens.TOKENIZER_PATTERN, common_pattern)
        indices = [i for sublist in gb['indices'].values for i in sublist]
        size = len(indices)
        return {'pattern': common_pattern,
                'sequence': sequence,
                'indices': indices,
                'cluster_size': size}



    def sequence_matcher(self, sequences):
        if len(sequences) > 1:
            pattern = sequences[0]
            for i in range(1, len(sequences)):
                matches = difflib.SequenceMatcher(None, pattern, sequences[i])
                m = [pattern[m.a:m.a + m.size] for m
                     in matches.get_matching_blocks() if m.size > 0]
                pattern = [val for sublist in m for val in sublist]
            return Tokens.detokenize_row(Tokens.TOKENIZER_PATTERN, pattern)
        else:
            return Tokens.detokenize_row(Tokens.TOKENIZER_PATTERN, sequences[0])



    def levenshtein_similarity(self, rows):
        """
        :param rows:
        :return:
        """
        if len(rows) > 1:
            return (
                [(1 - editdistance.eval(rows[0], rows[i]) / max(len(rows[0]), len(rows[i]))) for i in
                 range(0, len(rows))])
        else:
            return 1
