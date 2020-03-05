import nltk
import numpy as np
import difflib
from itertools import groupby


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
        if len(self.sequences) > 1:
            pattern = self.sequences[0]
            for i in range(1, len(self.sequences)):
                matches = difflib.SequenceMatcher(None, pattern, self.sequences[i])
                if add_placeholder:
                    m = [pattern[m.a:m.a + m.size] + ['(.*?)'] for m
                         in matches.get_matching_blocks() if m.size > 0]
                else:
                    m = [pattern[m.a:m.a + m.size] for m
                         in matches.get_matching_blocks() if m.size > 0]
                pattern = [val for sublist in m for val in sublist]
            return pattern
        else:
            return self.sequences[0]


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