from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from itertools import groupby
from collections import defaultdict


class Tokens(object):

    TOKENIZER = Tokenizer("conservative", spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    STOP = stopwords.words('english') + list(punctuation) + ["``", "''", u"\u2581"]

    def __init__(self, messages):

        self.messages = messages



    def process(self):
        """
        :return:
        """
        self.tokenized_pattern = self.pyonmttok(Tokens.TOKENIZER, self.messages)
        self.tokenized_cluster = self.clean_tokenized(self.tokenized_pattern)

        self.vocabulary_cluster = Tokens.get_vocabulary(self.tokenized_cluster)
        self.vocabulary_pattern = Tokens.get_vocabulary(self.tokenized_pattern)
        self.patterns = self.detokenize(self.tokenized_pattern)


    @staticmethod
    def remove_adjacent(L):
        return [elem for i, elem in enumerate(L) if i == 0 or L[i - 1] != elem]


    def detokenize(self, tokenized):
        return [self.TOKENIZER.detokenize([x for x, _ in groupby(row)]) for row in tokenized]


    @staticmethod
    def detokenize_row(tokenizer, row):
        remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i - 1] == '▁']
        row = [i for j, i in enumerate(row) if j not in remove_indices]
        return tokenizer.detokenize(Tokens.remove_adjacent(row))


    def remove_neighboring_duplicates(self, tokenized):
        n = []
        for row in tokenized:
            remove_indices = [i-1 for i, j in enumerate(row) if j == '｟*｠' and row[i-1] == '▁']
            row = [i for j, i in enumerate(row) if j not in remove_indices]
            n.append([x for x, _ in groupby(row)])
        return n


    @staticmethod
    def tokenize_string(tokenizer, row, clean=False):
        tokens, features = tokenizer.tokenize(row)
        if clean:
            return [i for i in tokens if i.lower() not in Tokens.STOP]
        else:
            return tokens


    def pyonmttok(self, tokenizer, strings):
        return [tokenizer.tokenize(line)[0] for line in strings]


    @staticmethod
    def clean_tokenized(tokenized):
        """
        Clean tokens from english stop words, numbers and punctuation
        :return:
        """
        return [[token for token in row if token.lower() not in Tokens.STOP] for row in tokenized]


    @staticmethod
    def clean_row(row):
        return [token for token in row if token.lower() not in Tokens.STOP]


    @staticmethod
    def get_vocabulary(tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))


    @staticmethod
    def get_term_frequencies(tokenized):
        frequency = defaultdict(int)
        for row in tokenized:
            for token in row:
                frequency[token] += 1
        return frequency