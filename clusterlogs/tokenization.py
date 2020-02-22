from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from itertools import groupby


class Tokens(object):

    TOKENIZER_CLUSTER = Tokenizer("conservative", spacer_annotate=False, preserve_placeholders=True)
    TOKENIZER_PATTERN = Tokenizer("conservative", spacer_annotate=True, preserve_placeholders=True)
    #TOKENIZER_PATTERN = Tokenizer("conservative", spacer_annotate=True, preserve_placeholders=True)

    def __init__(self, messages):

        self.messages = messages


    def process(self):
        """
        :return:
        """
        #self.tokenized_cluster = self.clean_tokens(self.pyonmttok(Tokens.TOKENIZER_CLUSTER, self.messages))
        #self.tokenized_pattern = self.remove_neighboring_duplicates(self.pyonmttok(Tokens.TOKENIZER_PATTERN, self.messages))
        self.tokenized_cluster = self.pyonmttok(Tokens.TOKENIZER_CLUSTER, self.messages)
        self.tokenized_pattern = self.pyonmttok(Tokens.TOKENIZER_PATTERN, self.messages)

        self.vocabulary_cluster = Tokens.get_vocabulary(self.tokenized_cluster)
        self.vocabulary_pattern = Tokens.get_vocabulary(self.tokenized_pattern)
        self.patterns = self.detokenize(self.tokenized_pattern)


    def detokenize(self, tokenized):
        return [self.TOKENIZER_PATTERN.detokenize([x for x, _ in groupby(row)]) for row in tokenized]


    @staticmethod
    def detokenize_row(tokenizer, row):
        return tokenizer.detokenize(row)
        # remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i-1] == '▁']
        # row = [i for j, i in enumerate(row) if j not in remove_indices]
        # return tokenizer.detokenize([x for x, _ in groupby(row)])


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
            stop = stopwords.words('english') + list(punctuation) + ["``", "''"]
            return [i for i in tokens if i.lower() not in stop ]
        else:
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
        return result


    @staticmethod
    def get_vocabulary(tokens):
        flat_list = [item for row in tokens for item in row]
        return list(set(flat_list))