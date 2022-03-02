from string import punctuation
from itertools import chain
from pyonmttok import Tokenizer
from collections import Counter
from nltk.corpus import stopwords
# import spacy
# from spacy import displacy
import re
# import nltk
# from nltk.tag.stanford import StanfordNERTagger
#
# st = StanfordNERTagger('/Users/maria/PycharmProjects/ClusterLog/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
#                        '/Users/maria/PycharmProjects/ClusterLog/stanford-ner-2020-11-17/stanford-ner.jar')


#NER = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

from itertools import groupby


# TOKENIZER = Tokenizer("space", spacer_annotate=True, preserve_placeholders=True, spacer_new=True)
STOP = stopwords.words('english') + list(punctuation) + ["``", "''", u"\u2581"]


def regexp_cleaning(messages):
    new_messages = []

    for row in messages:
        # replace date with DATE
        row = re.sub(r"(\d{2,4}[-|\s]?\d{2}[-|\s]?\d{2,4})", 'DATE', row)
        # replace time with TIME
        row = re.sub(r"\d{2}:\d{2}:\d{2}", 'TIME', row)
        # replace URLs
        row = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', row)
        # Removes words that contain at least one digit inside but not in the last place
        row = re.sub(r'([a-zA-Z_]*\d+[a-zA-Z_]*)+', '*', row)
        # replace numbers
        row = re.sub('(\d+)','*', row)

        # tokens = nltk.tokenize.word_tokenize(row)
        # tags = st.tag(tokens)
        # for tag in tags:
        #     if tag[1] == 'PERSON':
        #         print(tag)

        new_messages.append(row)

    return new_messages

# def ner_replacing(messages):
#     new_messages = []
#     for row in messages:
#         doc = NER(row)
#         for ent in doc.ents:
#             row = row.replace(ent.text, ent.label_)
#         new_messages.append(row)
#     return new_messages


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


def clean_tokenized(tokenized_messages, remove_stopwords=False):
    """
    This function removes tokens with any symbols other than letters.
    Optionally it can also remove any stop words.
    """
    if remove_stopwords:
        return [[token for token in row if token.lower() not in STOP and token.lower().isalpha()]
                for row in tokenized_messages]
    return [[token for token in row if token.lower().isalpha()]
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
