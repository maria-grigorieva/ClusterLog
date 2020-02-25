#!/usr/bin/python
import sys, getopt
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
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

    dct = Dictionary(tokenized)
    corpus = [dct.doc2bow(line) for line in tokenized]
    tfidf = TfidfModel(corpus, normalize=True)

    tfidf.save(outputfile)




if __name__ == "__main__":
    main(sys.argv[1:])