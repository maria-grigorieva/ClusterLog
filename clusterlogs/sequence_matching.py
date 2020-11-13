import random
import difflib
import numpy as np

from itertools import chain
from string import punctuation

from .utility import levenshtein_similarity_1_to_n


class Match:
    '''
    This class allows to extract the common pattern from a list of sequences.
    Create a new Match object for every pattern extraction task.
    '''
    def __init__(self, match_threshhold=0.8, add_placeholder=False):
        #self.sequences = sequences
        self.match_threshhold = match_threshhold
        self.add_placeholder = add_placeholder

    def sequence_matcher(self, sequences):
        unique = np.unique(sequences).tolist()
        if len(unique) <= 1:
            return unique[0]

        random.shuffle(unique)
        for x in unique:
            others = unique[:]
            others.remove(x)
            for sequence in others:
                matches = difflib.SequenceMatcher(None, x, sequence)
                if matches.ratio() < self.match_threshhold:
                    continue

                # We extract matching fragments of sequences
                # and change pattern to only contain those subsequences.
                # In the end this gives us a common part of all the sequences.
                match_ranges = matches.get_matching_blocks()[:-1]
                matches = [x[m.a:m.a + m.size] for m in match_ranges]
                if self.add_placeholder:  # Add a placeholder between matching subsequences
                    [match + ['(.*?)'] for match in matches]
                    matches[-1].pop()
                pattern = list(chain(*matches))  # concatenate inner lists

            if not pattern:
                continue
            junk = list(punctuation) + ['_', '(.*?)', '']
            # if at least one of the items in sequence is not junk - return True
            correct = any([token not in junk for token in pattern])
            return pattern if correct else x
        return x

    # This basically does the same as sequence_matcher, with a couple of differences:
    # * sequence_matcher picks only unique sequences
    # * SM picks random to compare to, this picks first one
    # * SM only compares sequences that have more similarity than self.match_threshhold

    # def matcher(self, sequences):
    #     x = sequences[0]
    #     for s in sequences:
    #         matches = difflib.SequenceMatcher(None, x, s)
    #         match_ranges = matches.get_matching_blocks()[:-1]
    #         matches = [x[m.a:m.a + m.size] for m in match_ranges]
    #         if self.add_placeholder:
    #             matches = [match + ['(.*?)'] for match in matches]
    #             matches[-1].pop()
    #         pattern = list(chain(*matches))  # concatenate inner lists
    #         junk = list(punctuation) + ['_', '(.*?)', '']
    #         # if at least one of the items in sequence is not junk - return True
    #         correct = any([token not in junk for token in pattern])
    #     return pattern if correct else x

    def matching_clusters(self, sequences, patterns):
        similarities = levenshtein_similarity_1_to_n(sequences)
        filtered, to_remove = [], []
        for i, value in enumerate(similarities):
            if value >= self.match_threshhold:
                filtered.append(sequences[i])
                to_remove.append(i)
        if not filtered:
            filtered.append(sequences[0])
            to_remove.append(0)
        patterns.append(self.matcher(filtered))
        sequences = np.delete(sequences, to_remove)
        if len(sequences) > 1:
            self.matching_clusters(sequences, patterns)
        elif len(sequences) == 1:
            patterns.append(sequences[0])
            np.delete(sequences, 0)

    def matrix_matching(self, sequences):
        if len(sequences) == 1:
            return sequences[0]
        else:
            x = list(map(list, zip(*sequences)))
            return [tokens[0] if len(tokens) == 1 else '(.*?)' for tokens in
                    [np.unique(line) for line in x]]
