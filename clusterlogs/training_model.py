#!/usr/bin/python
import sys, getopt
import re
from gensim.models import Word2Vec
from pyonmttok import Tokenizer


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
    messages = [line.rstrip('\n') for line in open(inputfile)]

    # Clean messages
   # messages_cleaned = [re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '*', item) for item in messages]
    messages_cleaned = cleaning(messages)

    # Tokenized cleaned messages
    tokenized = tokenization(messages_cleaned)

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


def tokenization(messages):
    tokenized = []
    for line in messages:
        #tokens, features = Tokenizer("conservative", spacer_annotate=True).tokenize(line)
        tokens, features = Tokenizer("conservative").tokenize(line)
        tokenized.append(tokens)
    return tokenized


def cleaning(messages):
    messages_cleaned = [0] * len(messages)
    for idx, item in enumerate(messages):
        item = re.sub(r'([ ])\1+', r'\1', item)
        item = re.sub(r'([* ])\1+', r'\1', item)
        # item = re.sub(r'((=)+( )*[0-9a-zA-Z_.|:;-]+)', '= {*}', item)
        # item = re.sub(r'((: )[0-9a-zA-Z_.|:;-]+)', ': {*}', item)
        item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', '{*}', item)
        messages_cleaned[idx] = item
    return messages_cleaned

if __name__ == "__main__":
    main(sys.argv[1:])