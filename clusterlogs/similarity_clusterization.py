import re
import pandas as pd

from .phraser import extract_common_phrases
from .sequence_matching import Match
from .tokenization import detokenize_row
from .utility import levenshtein_similarity_1_to_n


class SClustering:

    def __init__(self, groups, accuracy, add_placeholder, tokenizer_type, keywords_extraction):
        self.groups = groups
        self.accuracy = accuracy
        self.add_placeholder = add_placeholder
        self.tokenizer_type = tokenizer_type
        self.keywords_extraction = keywords_extraction

    def process(self):
        """
        Clustering of messages using sequence matching
        """
        result = []
        self.reclustering(self.groups.copy(deep=True), result)
        self.result = pd.DataFrame(result)
        return self.result.sort_values(by=['cluster_size'], ascending=False)

    def reclustering(self, df, result):
        """
        Clustering of the groups:
        - take the most frequent sequence of tokens and compare if with others
        - take all messages, which are similar with the 1st with more than accuracy threshold and
        join them into the new separate cluster
        - remove these messages from the initial group
        - repeat these steps while group has messages
        """
        # Detect the most frequent textual sequence
        top_sequence = df['sequence'].apply(tuple).describe().top
        # Calculate Levenshtein similarities between the most frequent and
        # all other textual sequences
        df['ratio'] = levenshtein_similarity_1_to_n(df['sequence'].values, top_sequence)
        # Filter the inistal DataFrame by accuracy values
        filtered = df[(df['ratio'] >= self.accuracy)]
        # Search common tokenized pattern and detokenize it
        pattern = Match(add_placeholder=self.add_placeholder)
        tokenized_pattern = pattern.sequence_matcher(filtered['tokenized_pattern'].values)
        textual_pattern = detokenize_row(tokenized_pattern, self.tokenizer_type)
        textual_pattern = re.sub(r'(\(\.\*\?\))(?:[\W\s]*\1)+', r'(.*?)', textual_pattern)
        # Search common sequence
        sequence = Match()
        common_sequence = sequence.sequence_matcher(filtered['sequence'].values)
        # Detect indices for the group
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        # Extract common phrases
        text = [' '.join(row) for row in filtered['sequence'].values]
        common_phrases = extract_common_phrases(text, self.keywords_extraction)

        result.append({'pattern': [textual_pattern],
                       'tokenized_pattern': tokenized_pattern,
                       'indices': indices,
                       'cluster_size': len(indices),
                       'sequence': common_sequence,
                       'common_phrases': common_phrases[:10]})

        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)
