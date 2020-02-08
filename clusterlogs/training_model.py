#!/usr/bin/python
import sys, getopt
import re
from gensim.models import Word2Vec
from pyonmttok import Tokenizer
from nltk.corpus import stopwords
from string import punctuation
from itertools import groupby


def main(argv):

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print
        'training_model.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    # Read messages from log file
    messages = [line for line in open(inputfile)]

    print('Log file contains {} lines'.format(len(messages)))

    # Clean messages
   # messages_cleaned = [re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '*', item) for item in messages]
    messages_cleaned = cleaning(messages)

    # Tokenized cleaned messages
    tokenized = tokenization(messages_cleaned)

    cleaned = remove_neighboring_duplicates(tokenized)

    try:
        word2vec = Word2Vec(cleaned,
                             size=300,
                             window=7,
                             min_count=1,
                             workers=4,
                             iter=10)

        word2vec.save(outputfile)
    except Exception as e:
        print('Training model error')


def tokenization(messages):
    tokenized = []
    for line in messages:
        #tokens, features = Tokenizer("conservative", spacer_annotate=True).tokenize(line)
        tokens, features = Tokenizer("conservative",
                                     spacer_annotate=False,
                                     preserve_placeholders=True)\
            .tokenize(line)
        tokenized.append(tokens)
    return tokenized


def cleaning(messages):
    messages_cleaned = [0] * len(messages)
    for idx, item in enumerate(messages):
        item = re.sub(r'([ ])\1+', r'\1', item)
        item = re.sub(r'([* ])\1+', r'\1', item)
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '｟*｠', item)
        messages_cleaned[idx] = item
    return messages_cleaned


def remove_neighboring_duplicates(tokenized):
    n = []
    for row in tokenized:
        remove_indices = [i - 1 for i, j in enumerate(row) if j == '｟*｠' and row[i - 1] == '▁']
        row = [i for j, i in enumerate(row) if j not in remove_indices]
        n.append([x for x, _ in groupby(row)])
    return n


def clean_tokens(tokenized):
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

if __name__ == "__main__":
    main(sys.argv[1:])