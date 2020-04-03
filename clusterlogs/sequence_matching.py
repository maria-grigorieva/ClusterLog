import nltk
import numpy as np
import difflib
from itertools import groupby
import random
from string import punctuation


class Match:

    def __init__(self, sequences):
        self.sequences = sequences


    def matcher(self):
        if len(self.sequences) > 1:
            fdist = nltk.FreqDist([i for l in self.sequences for i in l])
            #x = [token for token in lines[0] if (fdist[token] / len(lines) >= 1)]
            x = [token if (fdist[token]/len(self.sequences) >= 1) else '(.*?)' for token in self.sequences[0]]
            return [i[0] for i in groupby(x)]
        else:
            return self.sequences[0]


    def matrix_matching(self):
        x = list(map(list, zip(*self.sequences)))
        return [tokens[0] if len(tokens) == 1 else '(.*?)' for tokens in
                [np.unique(line) for line in x]]


    def sequence_matcher(self, add_placeholder=False):
        # number of attempts
        max_attempts = 3
        attempt = 1
        # detect unique sentences
        unique = np.unique(self.sequences)
        if len(unique) > 1:
            pattern = random.choice(unique)
            for i in range(1, len(unique)):
                matches = difflib.SequenceMatcher(None, pattern, unique[i])
                if matches.ratio() > 0.5:
                    m = [pattern[m.a:m.a + m.size] for m
                         in matches.get_matching_blocks() if m.size > 0]
                    pattern = [val for sublist in m for val in sublist]
                    if add_placeholder:
                        x = [item + ['(.*?)'] if i < len(m)-1 else item for i,item in enumerate(m)]
                        pattern = [val for sublist in x for val in sublist]
            # TODO:
            # if pattern is empty - try to make it based on another sample message
            is_empty = sum([True if token in list(punctuation) or token == '(.*?)' or
                                    token == '▁' else False for token in pattern])
            if is_empty == len(pattern) and attempt <= max_attempts:
                attempt += 1
                print('Search for common pattern for {}. Next attempt...'.format(pattern))
                self.sequence_matcher(add_placeholder)
            else:
                return pattern
        else:
            return unique[0]


    # def matching_clusters(self, sequences, patterns):
    #     if len(sequences) > 0:
    #         start = sequences[0]
    #         similarities = self.levenshtein_similarity(start, sequences)
    #         filtered = []
    #         to_remove = []
    #         for i, value in enumerate(similarities):
    #             if value >= 0.6:
    #                 filtered.append(sequences[i])
    #                 to_remove.append(i)
    #         sequences = np.delete(sequences, to_remove)
    #         patterns.append(self.sequence_matcher(filtered))
    #         while len(sequences) > 0:
    #             self.matching_clusters(sequences, patterns)
    #     else:
    #         patterns = sequences[0]
