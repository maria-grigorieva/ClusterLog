import pandas as pd
import sys, getopt
import csv

def main(argv):

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:t:", ["ifile=", "ofile=", "target="])
    except getopt.GetoptError:
        print
        'training_model.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -i <inputfile> -o <outputfile> -t <target_column>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-t", "--target"):
            target = arg

    with open(inputfile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        messages = [row[target] for row in reader]

    f = open(outputfile, 'w')
    for i in messages:
        f.write(i + '\n')
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])