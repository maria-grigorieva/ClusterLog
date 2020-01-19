from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, ToktokTokenizer, WordPunctTokenizer
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


    def process(self):
        """
        The best tokenizer for error messages is TreebankWordTokenizer (nltk).
        It's good at tokenizing file paths.
        Alternative tokenizer. It performs much faster, but worse in tokenizing of paths.
        It splits all paths by "/".
        TODO: This method should be optimized to the same tokenization quality as TreebankWordTokenizer
        :return:
        """
        self.tokenized_wordpunct = self.wordpunct()
        self.tokenized_pyonmttok = self.pyonmttok()


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

