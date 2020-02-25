#!/usr/bin/python
import sys, getopt
import numpy as np
from gensim.models import Word2Vec
from clusterlogs.tfidf import TermsAnalysis
from clusterlogs.tokenization import Tokens

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

    tokens = Tokens(messages)
    tokenized = Tokens.clean_tokenized(tokens.pyonmttok(Tokens.TOKENIZER, messages))
    vocab = tokens.get_vocabulary(tokenized)
    print('Initial vocabulary size is {}'.format(len(vocab)))
    tfidf = TermsAnalysis(tokenized)
    cleaned = tfidf.process()
    result = np.unique(cleaned)
    vocab = tokens.get_vocabulary(result)
    print('Cleaned vocabulary size is {}'.format(len(vocab)))

    try:
        word2vec = Word2Vec(np.unique(cleaned),
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