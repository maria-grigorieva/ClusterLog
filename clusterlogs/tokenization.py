from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint
from collections import OrderedDict
from itertools import groupby

TOKENS_LIMIT = 30

class Tokens(object):


    def __init__(self, messages):
        self.tokenizer_dbscan = Tokenizer("conservative", spacer_annotate=False, preserve_placeholders=True, segment_case=True)
        self.tokenizer_pattern = Tokenizer("conservative", spacer_annotate=True, preserve_placeholders=True, segment_case=True)
        self.hashed = None
        self.messages = messages


    def process(self):
        """
        :return:
        """
        self.tokenized_dbscan = self.remove_neighboring_duplicates(self.pyonmttok(self.tokenizer_dbscan, self.messages))
        self.tokenized_pattern = self.remove_neighboring_duplicates(self.pyonmttok(self.tokenizer_pattern, self.messages))
        # for item in self.tokenized_dbscan:
        #     print(item)
        #self.hashed = self.hashing(self.tokenized_dbscan)
        #self.vocabulary = self.get_vocabulary(self.tokenized)
        self.vocabulary_dbscan = self.get_vocabulary(self.tokenized_dbscan)
        self.vocabulary_pattern = self.get_vocabulary(self.tokenized_pattern)
        self.patterns = self.detokenize(self.tokenized_pattern)


    def detokenize(self, tokenized):
        return [self.tokenizer_pattern.detokenize(row) for row in tokenized]


    def remove_neighboring_duplicates(self, tokenized):
        n = []
        for row in tokenized:
            n.append([x for x, _ in groupby(row)])
        return n


    def tokenize_string(self, tokenizer, string):
        tokens, features = tokenizer.tokenize(string)
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


    def get_vocabulary(self, tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))