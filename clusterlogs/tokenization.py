from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from itertools import groupby
from collections import defaultdict


class Tokens(object):

    #TOKENIZER = Tokenizer("space", spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    STOP = stopwords.words('english') + list(punctuation) + ["``", "''", u"\u2581"]

    def __init__(self, messages, tokenizer_type):

        self.messages = messages
        Tokens.TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)


    def process(self):
        """
        :return:
        """
        self.tokenized = self.pyonmttok(Tokens.TOKENIZER, self.messages)


    @staticmethod
    def remove_adjacent(L):
        return [elem for i, elem in enumerate(L) if i == 0 or L[i - 1] != elem]


    def detokenize(self, tokenized):
        return [self.TOKENIZER.detokenize([x for x, _ in groupby(row)]) for row in tokenized]


    @staticmethod
    def detokenize_row(tokenizer, row):
        remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i - 1] == '▁']
        row = [i for j, i in enumerate(row) if j not in remove_indices]
        #return tokenizer.detokenize(Tokens.remove_adjacent(row))
        return tokenizer.detokenize(row)


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
        return [[token for token in row if token.lower().isalpha()]
                for row in tokenized]
        # return [[token for token in row if token.lower() not in Tokens.STOP and token.lower().isalpha()]
        #         for row in tokenized]


    @staticmethod
    def clean_row(row):
        return [token for token in row if token.lower() not in Tokens.STOP and token.lower().isalpha()]


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


    def to_lower(self, tokenized):
        return [[token.lower() for token in row] for row in tokenized]
