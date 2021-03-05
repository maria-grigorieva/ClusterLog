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
        # self.sequences = sequences
        self.match_threshhold = match_threshhold
        self.add_placeholder = add_placeholder

    def sequence_matcher(self, sequences):
        if len(sequences) <= 1:
            return sequences[0]

        # We have to do in this way because numpy.unique
        # does not preserve types of sequence members and can
        # transform an inner list into a string for some reason
        unique = list(set([tuple(x) for x in sequences]))
        unique = [list(x) for x in unique]
        random.shuffle(unique)

        for x in unique:
            pattern = []
            for sequence in unique:
                if x == sequence:
                    continue
                matches = difflib.SequenceMatcher(None, x, sequence)
                if matches.ratio() < self.match_threshhold:
                    continue

                # We extract matching fragments of sequences
                # and change pattern to only contain those subsequences.
                # In the end this gives us a common part of all the sequences.
                match_ranges = matches.get_matching_blocks()[:-1]
                matches = [x[m.a:m.a + m.size] for m in match_ranges]
                if self.add_placeholder:  # Add a placeholder between matching subsequences
                    placeholder_matches = []
                    # All the ifs in the following iteration are meant to
                    # keep several placeholders in a row from appearing
                    for match in matches:
                        if match == ['▁'] or (len(match) > 1 and match[-2:] == ['(.*?)', '▁']):
                            continue
                        if match[:2] == ['▁', '(.*?)']:
                            match = match[2:]
                        placeholder_matches.append(match + ['(.*?)'])
                    matches = placeholder_matches
                    # x.a + x.size == len(x) means that the ends of the patterns match
                    # so we don't need a placeholder at the end of the last match
                    if match_ranges[-1].a + match_ranges[-1].size == len(x):
                        matches[-1].pop()
                    # If only first word is different, there will be no placeholder without this if
                    if match_ranges[0].a != 0 or match_ranges[0].b != 0:
                        matches = [['(.*?)']] + matches
                pattern = list(chain(*matches))  # concatenate inner lists

            if not pattern:
                continue
            junk = list(punctuation) + ['▁', '(.*?)', '']
            # if at least one of the items in sequence is not junk - return True
            correct = any([token not in junk for token in pattern])
            return pattern if correct else x
        return x

    def matching_clusters(self, sequences, indices):
        if len(sequences) == 1:
            return [sequences[0]], [indices[0]]
        # Degree of similarity of every sequence to the first
        similarities = levenshtein_similarity_1_to_n(sequences)
        similar, others = [sequences[0]], []
        indices_sim, indices_other = [indices[0]], []
        for i, value in enumerate(similarities):
            if value >= self.match_threshhold:
                similar.append(sequences[i + 1])
                indices_sim.append(indices[i + 1])
            else:
                others.append(sequences[i + 1])
                indices_other.append(indices[i + 1])
        patterns = [self.sequence_matcher(similar)]
        final_indices = [[i for sublist in indices_sim for i in sublist]]
        if len(others) > 1:
            res_patterns, res_indices = self.matching_clusters(others, indices_other)
            patterns.extend(res_patterns)
            final_indices.extend(res_indices)

        # We need this elif instead of an if in the beginning
        # to check for messages being the same after sequence matching
        elif len(others) == 1 and others[0] not in patterns:
            patterns.append(others[0])
            final_indices.append(indices_other[0])
        return patterns, final_indices

    def matrix_matching(self, sequences):
        if len(sequences) == 1:
            return sequences[0]
        else:
            x = list(map(list, zip(*sequences)))
            return [tokens[0] if len(tokens) == 1 else '(.*?)' for tokens in
                    [np.unique(line) for line in x]]
