import random
import difflib
import numpy as np

from itertools import chain
from string import punctuation


class Match:
    '''
    This class allows to extract the common pattern from a list of sequences.
    Create a new Match object for every pattern extraction task.
    '''
    def __init__(self, sequences, match_threshhold=0.5, max_attempts=3):
        self.sequences = sequences
        self.match_threshhold = match_threshhold
        self.max_attempts = max_attempts
        self.attempt_number = 1

    def sequence_matcher(self, add_placeholder=False):
        unique = np.unique(self.sequences)
        if len(unique) <= 1:
            return unique[0]

        pattern = random.choice(unique)
        for sequence in unique:
            matches = difflib.SequenceMatcher(None, pattern, sequence)
            if matches.ratio() < self.match_threshhold:
                continue

            # We extract matching portions of sequences
            # and change pattern to only contain those subsequences.
            # In the end this gives us a common part of all the sequences.
            match_ranges = matches.get_matching_blocks()[:-1]
            matches = [pattern[m.a:m.a + m.size] for m in match_ranges]
            if add_placeholder:  # Add a placeholder between matching subsequences
                matches = [match + ['(.*?)'] for match in matches]
                matches[-1].pop()
            pattern = list(chain(*matches))  # concatenate inner lists

        # TODO: if pattern is empty - try to make it based on another sample message
        junk = list(punctuation) + ['_', '(.*?)', '']
        # if at least one of the items in sequence is not junk - return True
        correct = any([token not in junk for token in pattern])
        if correct or self.attempt_number > self.max_attempts:
            return pattern
        else:
            self.attempt_number += 1
            print('Search for common pattern for {}. Next attempt...'.format(pattern))
            self.sequence_matcher(add_placeholder)

#    def matcher(self):
#        if len(self.sequences) > 1:
#            fdist = nltk.FreqDist([i for l in self.sequences for i in l])
#            # x = [token for token in lines[0] if (fdist[token] / len(lines) >= 1)]
#            x = [token if (fdist[token] / len(self.sequences) >= 1) else '(.*?)' for token in self.sequences[0]]
#            return [i[0] for i in groupby(x)]
#        else:
#            return self.sequences[0]

#    def matrix_matching(self):
#        x = list(map(list, zip(*self.sequences)))
#        return [tokens[0] if len(tokens) == 1 else '(.*?)' for tokens in
#                [np.unique(line) for line in x]]

#    def matching_clusters(self, sequences, patterns):
#        if len(sequences) > 0:
#            start = sequences[0]
#            similarities = self.levenshtein_similarity(start, sequences)
#            filtered = []
#            to_remove = []
#            for i, value in enumerate(similarities):
#                if value >= 0.6:
#                    filtered.append(sequences[i])
#                    to_remove.append(i)
#            sequences = np.delete(sequences, to_remove)
#            patterns.append(self.sequence_matcher(filtered))
#            while len(sequences) > 0:
#                self.matching_clusters(sequences, patterns)
#        else:
#            patterns = sequences[0]
