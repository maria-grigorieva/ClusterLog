import pandas as pd
import editdistance
from .tokenization import *
import pprint
from .phraser import phraser
from .sequence_matching import Match
import re


class SClustering:

    def __init__(self, groups, accuracy, add_placeholder, tokenizer_type):
        self.groups = groups
        self.accuracy = accuracy
        self.add_placeholder = add_placeholder
        self.tokenizer_type = tokenizer_type


    def process(self):
        """
        Clusterization messages using sequence matching
        :param df:
        :param accuracy:
        :return:
        """
        result = []
        self.reclustering(self.groups.copy(deep=True), result)
        self.result = pd.DataFrame(result)
        return self.result.sort_values(by=['cluster_size'], ascending=False)



    def reclustering(self, df, result):
        """
        Clusterization of the groups:
        - take the most frequent sequence of tokens and compare if with others
        - take all messages, which are similar with the 1st with more than accuracy threshold and
        join them into the new separate cluster
        - remove these messages from the initial group
        - repeat these steps while group has messages
        :param df:
        :param result:
        :param accuracy:
        :return:
        """
        # Detect the most frequent textual sequence
        top_sequence = df['sequence'].apply(tuple).describe().top
        # Calculate Levenshtein similarities between the most frequent and
        # all other textual sequences
        df['ratio'] = self.levenshtein_similarity(top_sequence, df['sequence'].values)
        # Filter the inistal DataFrame by accuracy values
        filtered = df[(df['ratio'] >= self.accuracy)]
        # Search common tokenized pattern and detokenize it
        pattern = Match(filtered['tokenized_pattern'].values)
        tokenized_pattern = pattern.sequence_matcher(self.add_placeholder)
        textual_pattern = detokenize_row(tokenized_pattern, self.tokenizer_type)
        textual_pattern = re.sub('\((.*?)\)+[\S\s]*\((.*?)\)+', '(.*?)', textual_pattern)
        # print(tokenized_pattern)
        # Search common sequence
        sequence = Match(filtered['sequence'].values)
        common_sequence = sequence.sequence_matcher(False)
        # Detect indices for the group
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        # Convert list of sequences to text
        text = '. '.join([' '.join(row) for row in filtered['sequence'].values])
        # Extract common phrases
        phrases_pyTextRank = phraser(text, 'pyTextRank')
        phrases_RAKE = phraser(text, 'RAKE')

        result.append({'pattern': [textual_pattern],
                       'tokenized_pattern': tokenized_pattern,
                       'indices': indices,
                       'cluster_size': len(indices),
                       'sequence': common_sequence,
                       'common_phrases_pyTextRank': phrases_pyTextRank.extract_common_phrases(),
                       'common_phrases_RAKE': phrases_RAKE.extract_common_phrases()})

        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)



    def levenshtein_similarity(self, top, rows):
        """
        Search similarities between top and all other sequences of tokens.
        May be used for strings as well.
        top - most frequent sequence
        rows - all sequences
        :param rows:
        :return:
        """
        if len(rows) > 1 and len(top) > 0:
            try:
                return (
                    [(1 - editdistance.eval(top, rows[i]) / max(len(top), len(rows[i]))) for i in
                     range(0, len(rows))])
            except Exception as e:
                print(rows)
                print(top)
        else:
            return 1


