import pandas as pd
import sys
sys.path.append('..')
import pprint
from clusterlogs.pipeline import Chain

def main():

    # CREATE MODE
    # df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/data_sample30days.csv', sep=';')
    # target = 'message'
    # cluster = Chain(df, target, mode='create', model_name='../models/test.model',
    #                          matching_accuracy=0.8, clustering_type='dbscan', output_type='html',
    #                  output_fname='../reports/test_dbscan_p', keywords_extraction='rake_nltk')
    # cluster.process()


    # UPDATE MODE
    df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/harvester_errors24.csv', sep=';')
    target = 'message'
    cluster = Chain(df, target, mode='process', model_name='../models/harvester_30days.model',
                     matching_accuracy=0.5, clustering_type='dbscan', output_type='html',
                     output_fname='../reports/test_dbscan_p', keywords_extraction='rake_nltk')
    cluster.process()



if __name__ == "__main__":
    sys.exit(main())

