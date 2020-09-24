import random
import difflib
import numpy as np

from itertools import chain, groupby
from string import punctuation
import editdistance
import nltk



class Match:
    '''
    This class allows to extract the common pattern from a list of sequences.
    Create a new Match object for every pattern extraction task.
    '''
    def __init__(self, sequences, match_threshhold=0.8, max_attempts=10, add_placeholder=False):
        self.sequences = sequences
        self.match_threshhold = match_threshhold
        self.max_attempts = max_attempts
        self.attempt_number = 1
        self.add_placeholder = add_placeholder

    def sequence_matcher(self):
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
            if self.add_placeholder:  # Add a placeholder between matching subsequences
                [match + ['(.*?)'] for match in matches]
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
            self.sequence_matcher()


    def matcher(self, sequences):
        pattern = sequences[0]
        for s in sequences:
            matches = difflib.SequenceMatcher(None, pattern, s)
            match_ranges = matches.get_matching_blocks()[:-1]
            matches = [pattern[m.a:m.a + m.size] for m in match_ranges]
            if self.add_placeholder:
                matches = [match + ['(.*?)'] for match in matches]
                matches[-1].pop()
            pattern = list(chain(*matches))  # concatenate inner lists
        return pattern


    def matching_clusters(self, sequences, patterns):
        start = sequences[0]
        similarities = Match.levenshtein_similarity(start, sequences)
        filtered, to_remove = [], []
        for i, value in enumerate(similarities):
            if value >= 0.7:
                filtered.append(sequences[i])
                to_remove.append(i)
        patterns.append(self.matcher(filtered))
        sequences = np.delete(sequences, to_remove)
        if len(sequences) > 1:
            self.matching_clusters(sequences, patterns)
        elif len(sequences) == 1:
            patterns.append(sequences[0])
            np.delete(sequences, 0)


    @staticmethod
    def levenshtein_similarity(top, rows):
        """
        Search similarities between top and all other sequences of tokens.
        May be used for strings as well.
        top - most frequent sequence
        rows - all sequences
        """
        if len(rows) > 1 and len(top) > 0:
            try:
                return (
                    [(1 - editdistance.eval(top, rows[i]) / max(len(top), len(rows[i]))) for i in
                     range(0, len(rows))])
            except Exception:
                print(rows)
                print(top)
        else:
            return 1


    def matrix_matching(self, sequences):
        if len(sequences) == 1:
            return sequences[0]
        else:
            x = list(map(list, zip(*sequences)))
            return [tokens[0] if len(tokens) == 1 else '(.*?)' for tokens in
                    [np.unique(line) for line in x]]


    # def matcher(self, sequences):
    #    if len(sequences) > 1:
    #        fdist = nltk.FreqDist([token for row in sequences for token in row])
    #        # x = [token for token in lines[0] if (fdist[token] / len(lines) >= 1)]
    #        x = [token if (fdist[token] / len(sequences) >= 1) else '(.*?)' for token in sequences[0]]
    #        print(x)
    #        print([i[0] for i in groupby(x)])
    #        return [i[0] for i in groupby(x)]
    #    else:
    #        return sequences[0]


