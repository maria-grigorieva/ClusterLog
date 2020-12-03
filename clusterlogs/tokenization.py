from string import punctuation
from itertools import chain
from pyonmttok import Tokenizer
from collections import Counter
from nltk.corpus import stopwords

from itertools import groupby


# TOKENIZER = Tokenizer("space", spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
STOP = stopwords.words('english') + list(punctuation) + ["``", "''", u"\u2581"]


def tokenize_messages(messages, tokenizer_type, spacer_annotate=True, spacer_new=True):
    tokenizer = Tokenizer(tokenizer_type, spacer_annotate=spacer_annotate, preserve_placeholders=True, spacer_new=spacer_new)
    return [tokenizer.tokenize(line)[0] for line in messages]


def detokenize_row(row, tokenizer_type):
    tokenizer = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    remove_indices = [i - 1 for i, token in enumerate(row) if token == '｟*｠' and row[i - 1] == '▁']
    row = [token for i, token in enumerate(row) if i not in remove_indices]
    # return tokenizer.detokenize(Tokens.remove_adjacent(row))
    return tokenizer.detokenize(row)


def get_vocabulary(tokenized_messages):
    '''
    A function that, given a list of token sequences,
    returns a list of tokens unique to all the sequences
    '''
    return list(set(chain(*tokenized_messages)))


def get_term_frequencies(tokenized_messages):
    '''
    A function that, given a list of token sequences,
    returns a dictionary with number of occurences for each token
    '''
    frequency = Counter(chain(*tokenized_messages))
    return dict(frequency)


def to_lower(tokenized_messages):
    '''
    A function that makes every token lowercase
    '''
    return [[token.lower() for token in row] for row in tokenized_messages]


# def clean_tokenized(tokenized_messages, remove_stopwords=False):
#     """
#     This function removes tokens with any symbols other than letters.
#     Optionally it can also remove any stop words.
#     """
    
#     if remove_stopwords:
#         return [[token for token in row if token.lower() not in STOP and token.lower().isalpha()]
#                 for row in tokenized_messages]
#     return [[token for token in row if token.lower().isalpha()]
#             for row in tokenized_messages]

def clean_tokenized(tokenized_messages):
    """
    This function removes tokens appearing once and stopwords
    """
    frequency=get_term_frequencies(tokenized_messages)
            
    stoplist = ['the','with','a','an','but','of','on','to','all','has','have','been','for','in','it','its','itself',
                'this','that','those','these','is','are','were','was','be','being','having','had','does','did','doing',
                'and','if','about','again','then','so','too','cern','cms','atlas','by','srm','ifce', 'err']
    
    return [[token for token in row if token not in stoplist and frequency[token] != 1]
            for row in tokenized_messages]

# def remove_adjacent(L):
#     return [elem for i, elem in enumerate(L) if i == 0 or L[i - 1] != elem]


def detokenize_messages(tokenized_messages, tokenizer_type):
    TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
    return [TOKENIZER.detokenize([x for x, _ in groupby(row)]) for row in tokenized_messages]


# def remove_neighboring_duplicates(tokenized):
#     n = []
#     for row in tokenized:
#         remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i - 1] == '▁']
#         row = [i for j, i in enumerate(row) if j not in remove_indices]
#         n.append([x for x, _ in groupby(row)])
#     return n


# def tokenize_string(row, tokenizer_type, clean=False):
#     TOKENIZER = Tokenizer(tokenizer_type, spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
#     tokens, features = TOKENIZER.tokenize(row)
#     if clean:
#         return [i for i in tokens if i.lower() not in STOP]
#     else:
#         return tokens


# def clean_row(row):
#     return [token for token in row if token.lower() not in STOP and token.lower().isalpha()]
