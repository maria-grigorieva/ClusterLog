from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, ToktokTokenizer, WordPunctTokenizer
from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
import pprint

class Tokens(object):

    def __init__(self, messages, type='nltk'):
        self.messages = messages
        self.type = type
        self.tokenized_wordpunct = []
        self.tokenized_treebank = []


    def process(self):
        """
        The best tokenizer for error messages is TreebankWordTokenizer (nltk).
        It's good at tokenizing file paths.
        Alternative tokenizer. It performs much faster, but worse in tokenizing of paths.
        It splits all paths by "/".
        TODO: This method should be optimized to the same tokenization quality as TreebankWordTokenizer
        :return:
        """
        if self.type == 'nltk':
            for line in self.messages:
                self.tokenized_wordpunct.append(WordPunctTokenizer().tokenize(line))
                self.tokenized_treebank.append(TreebankWordTokenizer().tokenize(line))
        # elif self.tokenizer == 'pyonmttok':
        #     tokenizer = Tokenizer("space", joiner_annotate=False, segment_numbers=False)
        #     for line in self.messages:
        #         tokens, features = tokenizer.tokenize(line)
        #         tokenized.append(tokens)
        self.tokenized_wordpunct = self.clean_tokens(self.tokenized_wordpunct)


    def clean_tokens(self, tokenized):
        """
        Clean tokens from english stop words, numbers and punctuation
        :return:
        """
        stop = stopwords.words('english') + list(punctuation) + ["``", "''"]
        cleaned_tokens = []
        for row in tokenized:
            cleaned_tokens.append([i for i in row if i.lower() not in stop])
        return cleaned_tokens


    def get_vocabulary(self):
        flat_list = [item for row in self.tokenized_wordpunct for item in row]
        self.vocabulary = set(flat_list)
        return self.vocabulary

