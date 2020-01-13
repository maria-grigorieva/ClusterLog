from nltk.tokenize import TreebankWordTokenizer
from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from urllib.parse import urlparse


class Tokens(object):

    def __init__(self, messages, type='nltk'):
        self.messages = messages
        self.type = type


    def process(self):
        """
        The best tokenizer for error messages is TreebankWordTokenizer (nltk).
        It's good at tokenizing file paths.
        Alternative tokenizer. It performs much faster, but worse in tokenizing of paths.
        It splits all paths by "/".
        TODO: This method should be optimized to the same tokenization quality as TreebankWordTokenizer
        :return:
        """
        tokenized = []
        if self.type == 'nltk':
            for line in self.messages:
                tokenized.append(TreebankWordTokenizer().tokenize(line))
        elif self.tokenizer == 'pyonmttok':
            tokenizer = Tokenizer("space", joiner_annotate=False, segment_numbers=False)
            for line in self.messages:
                tokens, features = tokenizer.tokenize(line)
                tokenized.append(tokens)
        self.tokenized = self.clean_tokens(tokenized)
        return self.tokenized


    def clean_tokens(self, tokenized):
        """
        Clean tokens from english stop words, numbers and punctuation
        :return:
        """
        stop = stopwords.words('english') + list(punctuation) + ["``", "''"]
        cleaned_tokens = []
        for row in tokenized:
            cleaned_tokens.append([i for i in row if i.lower() not in stop and not i.lower().isnumeric()])
        return cleaned_tokens


    def get_vocabulary(self):
        flat_list = [item for row in self.tokenized for item in row]
        self.vocabulary = set(flat_list)
        return self.vocabulary

    #
    # def getPathTokens(full_path):
    #     print('printPathTokens() called: %s' % full_url)
    #
    #     p_full = urlparse(full_url).path
    #
    #     print(' . p_full url: %s' % p_full)
    #
    #     # Split the path using rpartition method of string
    #     # rpartition "returns a tuple containing the part the before separator,
    #     # argument string and the part after the separator"
    #     (rp_left, rp_match, rp_right) = p_full.rpartition('/')
    #
    #     if rp_match == '':  # returns the rpartition separator if found
    #         print(' . No slashes found in path')
    #     else:
    #         print(' . path to last resource: %s' % rp_left)
    #         if rp_right == '':  # Ended with a slash
    #             print(' . last resource: (none)')
    #         else:
    #             print(' . last resource: %s' % (rp_right))
