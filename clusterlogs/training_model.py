#!/usr/bin/python
import sys, getopt
import numpy as np
from gensim.models import Word2Vec
from clusterlogs.data_preparation import *
from clusterlogs.tokenization import *

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
    cleaned_strings = clean_messages(messages)
    unique = np.unique(cleaned_strings)
    tokenized = tokenize_messages(unique, 'space')
    print('Number of unique lines after cleaning is {}'.format(len(unique)))
    # cleaned = [[token for token in row if token != u"\u2581"] for row in tokens.tokenized]

    try:
        word2vec = Word2Vec(tokenized,
                             size=300,
                             window=7,
                             min_count=1,
                             workers=4,
                             iter=10)

        word2vec.save(outputfile)
    except Exception as e:
        print('Training model error')


if __name__ == "__main__":
    main(sys.argv[1:])