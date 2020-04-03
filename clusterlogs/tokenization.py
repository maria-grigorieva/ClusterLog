from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from itertools import groupby
from collections import defaultdict


#TOKENIZER = Tokenizer("space", spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
STOP = stopwords.words('english') + list(punctuation) + ["``", "''", u"\u2581"]

def remove_adjacent(L):
    return [elem for i, elem in enumerate(L) if i == 0 or L[i - 1] != elem]


def detokenize(tokenized_messages, tokenizer_type):
    TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    return [TOKENIZER.detokenize([x for x, _ in groupby(row)]) for row in tokenized_messages]


def detokenize_row(row, tokenizer_type):
    TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i - 1] == '▁']
    row = [i for j, i in enumerate(row) if j not in remove_indices]
    #return tokenizer.detokenize(Tokens.remove_adjacent(row))
    return TOKENIZER.detokenize(row)


def remove_neighboring_duplicates(tokenized):
    n = []
    for row in tokenized:
        remove_indices = [i-1 for i, j in enumerate(row) if j == '｟*｠' and row[i-1] == '▁']
        row = [i for j, i in enumerate(row) if j not in remove_indices]
        n.append([x for x, _ in groupby(row)])
    return n


def tokenize_string(row, tokenizer_type, clean=False):
    TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    tokens, features = TOKENIZER.tokenize(row)
    if clean:
        return [i for i in tokens if i.lower() not in STOP]
    else:
        return tokens


def tokenize_messages(messages, tokenizer_type):
    TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    return [TOKENIZER.tokenize(line)[0] for line in messages]


def clean_tokenized(tokenized_messages):
    """
    Clean tokens from english stop words, numbers and punctuation
    :return:
    """
    return [[token for token in row if token.lower().isalpha()]
            for row in tokenized_messages]
    # return [[token for token in row if token.lower() not in Tokens.STOP and token.lower().isalpha()]
    #         for row in tokenized]


def clean_row(row):
    return [token for token in row if token.lower() not in STOP and token.lower().isalpha()]


def get_vocabulary(tokens):
    flat_list = [item for row in tokens for item in row]
    return list(set(flat_list))



def get_term_frequencies(tokenized_messages):
    frequency = defaultdict(int)
    for row in tokenized_messages:
        for token in row:
            frequency[token] += 1
    return frequency


def to_lower(tokenized_messages):
    return [[token.lower() for token in row] for row in tokenized_messages]
