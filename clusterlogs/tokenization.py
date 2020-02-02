from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint
from pyonmttok import Tokenizer

TOKENS_LIMIT = 30

class Tokens(object):


    def __init__(self, messages):
        self.tokenizer = Tokenizer("conservative", spacer_annotate=True)
        self.messages = messages
        self.tokenized = None
        self.tokenized_cleaned = None
        self.vocabulary = None
        self.vocabulary_cleaned = None


    def process(self):
        """
        :return:
        """
        #self.tokenized = self.pyonmttok(self.messages)
        self.tokenized = self.pyonmttok(self.messages)
        #self.vocabulary = self.get_vocabulary(self.tokenized)
        self.vocabulary = self.get_vocabulary(self.tokenized)


    def tokenize_string(self, string):
        tokens, features = self.tokenizer.tokenize(string)
        return tokens


    def pyonmttok(self, strings):
        tokenized = []
        for line in strings:
            tokens, features = self.tokenizer.tokenize(line)
            tokenized.append(tokens)
        return tokenized



    def clean_tokens(self, tokenized):
        """
        Clean tokens from english stop words, numbers and punctuation
        :return:
        """
        stop = stopwords.words('english') + list(punctuation) + ["``", "''"]
        return [i for row in tokenized for i in row if i.lower() not in stop]



    def get_vocabulary(self, tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))

