#!/usr/bin/python
import sys
import getopt
from gensim.models import Word2Vec
from clusterlogs.data_preparation import alpha_cleaning, clean_messages
import pprint
from clusterlogs.tokenization import get_term_frequencies


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
    pprint.pprint("First 10 messages: ")
    pprint.pprint(messages[0:10])

    print('Log file contains {} lines'.format(len(messages)))

    cleaned_strings = clean_messages(messages)
    tokenized = [row.split(' ') for row in cleaned_strings]

    # get frequence of cleaned tokens
    frequency = get_term_frequencies(tokenized)
    # remove tokens that appear only once and save tokens which are textual substrings
    tokenized = [
        [token for token in row if frequency[token] > 1]
        for row in tokenized]

    pprint.pprint("First 100 cleaned messages: ")
    pprint.pprint(cleaned_strings[0:100])
    # unique = np.unique(cleaned_strings)
    # print('Number of unique lines after cleaning is {}'.format(len(unique)))
    # tokenized = [row.split(' ') for row in unique]

    #tokenized = tokenize_messages(unique, 'space', spacer_annontate=False, spacer_new=False)
    #print(tokenized)

    print('Messages has been tokenized')

    try:
        word2vec = Word2Vec(tokenized,
                            size=300,
                            window=7,
                            min_count=10,
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
