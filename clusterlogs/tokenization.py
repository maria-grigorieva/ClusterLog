from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint
from collections import OrderedDict
from itertools import groupby

TOKENS_LIMIT = 30

class Tokens(object):

    tokenizer_dbscan = Tokenizer("conservative", spacer_annotate=False, preserve_placeholders=True)
    tokenizer_pattern = Tokenizer("conservative", spacer_annotate=True, preserve_placeholders=True)

    def __init__(self, messages):

        self.messages = messages


    def process(self):
        """
        :return:
        """
        self.tokenized_dbscan = self.remove_neighboring_duplicates(self.pyonmttok(Tokens.tokenizer_dbscan, self.messages))
        self.tokenized_pattern = self.remove_neighboring_duplicates(self.pyonmttok(Tokens.tokenizer_pattern, self.messages))
        self.vocabulary_dbscan = Tokens.get_vocabulary(self.tokenized_dbscan)
        self.vocabulary_pattern = Tokens.get_vocabulary(self.tokenized_pattern)
        self.patterns = self.detokenize(self.tokenized_pattern)


    def detokenize(self, tokenized):
        return [self.tokenizer_pattern.detokenize([x for x, _ in groupby(row)]) for row in tokenized]

    @staticmethod
    def detokenize_row(tokenizer, row):
        remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i-1] == '▁']
        row = [i for j, i in enumerate(row) if j not in remove_indices]
        return tokenizer.detokenize([x for x, _ in groupby(row)])


    def remove_neighboring_duplicates(self, tokenized):
        n = []
        for row in tokenized:
            remove_indices = [i-1 for i, j in enumerate(row) if j == '｟*｠' and row[i-1] == '▁']
            row = [i for j, i in enumerate(row) if j not in remove_indices]
            n.append([x for x, _ in groupby(row)])
        return n


    @staticmethod
    def tokenize_string(tokenizer, row):
        tokens, features = tokenizer.tokenize(row)
        return tokens


    def pyonmttok(self, tokenizer, strings):
        tokenized = []
        for line in strings:
            tokens, features = tokenizer.tokenize(line)
            tokenized.append(tokens)
        return tokenized



    def clean_tokens(self, tokenized):
        """
        Clean tokens from english stop words, numbers and punctuation
        :return:
        """
        stop = stopwords.words('english') + list(punctuation) + ["``", "''"]
        result = []
        for row in tokenized:
            tokenized = []
            for i in row:
                if i.lower() not in stop:
                    tokenized.append(i)
            result.append(tokenized)
            #print(tokenized)
        return result

    def hashing(self, tokenized):
        return [hash(tuple(row)) for row in tokenized]

    @staticmethod
    def get_vocabulary(tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))