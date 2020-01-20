from nltk.tokenize import WordPunctTokenizer
from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint

class Tokens(object):

    def __init__(self, messages):
        self.messages = messages
        self.type = type
        self.tokenized_wordpunct = []
        self.tokenized_pyonmttok = []
        self.wordpunct_vocab = None
        self.pyonmttok_vocab = None


    def process(self):
        """
        :return:
        """
        self.tokenized_wordpunct = self.wordpunct()
        self.tokenized_pyonmttok = self.pyonmttok()
        self.wordpunct_vocab = self.get_vocabulary(self.tokenized_wordpunct)
        self.pyonmttok_vocab = self.get_vocabulary(self.tokenized_pyonmttok)


    def wordpunct(self):

        return [WordPunctTokenizer().tokenize(line) for line in self.messages]


    def pyonmttok(self):
        tokenizer = Tokenizer("space", joiner_annotate=False, segment_numbers=False)
        tokenized = []
        for line in self.messages:
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

