#!/usr/bin/python
import sys
import getopt
from gensim.models import Word2Vec
from clusterlogs.data_preparation import clean_messages
import pprint
import numpy as np
import pandas as pd

from clusterlogs.tokenization import tokenize_messages

from utility import gather_df
from utility import parallel_file_read

comm = None
comm_size = 1
comm_rank = 0

import os
if os.environ.get("USE_MPI"):
    import pandas as pd
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()


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
    messages = [line[:-1] for line in parallel_file_read(comm, inputfile)]
    print(messages[0:500])

    # if comm_rank == 0:
    #     pprint.pprint("First 10 messages: ")
    #     pprint.pprint(messages[0:10])

    print('Log file contains {} lines'.format(len(messages)))

    cleaned_strings = clean_messages(messages)
    pprint.pprint(cleaned_strings)
    unique = np.unique(cleaned_strings)
    pprint.pprint("Unique: ")
    print(len(unique))

    tokenized = tokenize_messages(unique, 'space', spacer_annotate=False, spacer_new=False)

    # tokenized = tokenize_messages(messages, 'space', spacer_annotate=False, spacer_new=False)

    print('Messages has been tokenized')

    # tokenized = gather_df(comm, pd.DataFrame(tokenized).head(500)).values.tolist()

    # print(tokenized)
    #
    if comm_rank == 0:
        try:
            word2vec = Word2Vec(tokenized,
                                size=300,
                                window=7,
                                min_count=1,
                                workers=4,
                                iter=30)

            word2vec.save(outputfile)

            print('Training has finished. Model saved in file.')
        except Exception as e:
            print('Training model error:', e)


if __name__ == "__main__":
    main(sys.argv[1:])
