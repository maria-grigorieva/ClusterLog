from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint

TOKENS_LIMIT = 30

class Tokens(object):


    def __init__(self, cleaned, messages):
        self.cleaned = cleaned
        self.messages = messages
        self.type = type
        self.tokenized = None
        self.tokenized_cleaned = None
        self.vocabulary = None
        self.vocabulary_cleaned = None


    def process(self):
        """
        :return:
        """
        self.tokenized = self.pyonmttok(self.messages)
        self.tokenized_cleaned = self.pyonmttok(self.cleaned)
        self.vocabulary = self.get_vocabulary(self.tokenized)
        self.vocabulary_cleaned = self.get_vocabulary(self.tokenized_cleaned)


    def pyonmttok(self, strings):
        tokenizer = Tokenizer("conservative", spacer_annotate=True)
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
        return [i for row in tokenized for i in row if i.lower() not in stop]



    def get_vocabulary(self, tokens):
        flat_list = [item for row in tokens for item in row]
        self.vocabulary = set(flat_list)
        return self.vocabulary

