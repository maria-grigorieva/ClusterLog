#!/usr/bin/python
import sys
import getopt
import numpy as np

from gensim.models import Word2Vec

from clusterlogs.tokenization import tokenize_messages
from clusterlogs.data_preparation import clean_messages,alpha_cleaning
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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

    cleaned_strings = alpha_cleaning(messages)
    unique = np.unique(cleaned_strings)
    # print(unique)
    # tokenized = tokenize_messages(unique, 'space')
    print('Number of unique lines after cleaning is {}'.format(len(unique)))
    # cleaned = [[token for token in row if token != u"\u2581"] for row in tokens.tokenized]
    tokenized = [row.split(' ') for row in unique]

    #tokenized = tokenize_messages(unique, 'space', spacer_annontate=False, spacer_new=False)
    print(tokenized)

    print('Messages has been tokenized')

    try:
        word2vec = Word2Vec(tokenized,
                            size=300,
                            window=7,
                            min_count=2,
                            workers=4,
                            iter=30)

        word2vec.save(outputfile)

        # tagged_docs = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(tokenized)]
        # doc2vec = Doc2Vec(tagged_docs, vector_size=200,
        #                        window=7, workers=4,
        #                        min_count=2, epochs=10)
        #
        # doc2vec.save(outputfile)
        print('Training has finished. Model saved in file. Thanks for coming :)')
    except Exception as e:
        print('Training model error:', e)


if __name__ == "__main__":
    main(sys.argv[1:])
