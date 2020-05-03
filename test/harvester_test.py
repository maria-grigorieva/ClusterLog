import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():

    # CREATE MODE
    # df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/data_sample30days.csv', sep=';')
    # target = 'message'
    # cluster = pipeline.Chain(df, target, mode='create', model_name='../models/harvester_30days.model',
    #                          matching_accuracy=0.8,
    #                          clustering_type='ML', output_file='../reports/harvester_create.html')
    # cluster.process()


    # UPDATE MODE
    df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/harvester_errors24.csv', sep=';')
    target = 'message'
    cluster = pipeline.Chain(df, target, mode='update', model_name='../models/harvester_30days.model',matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/harvester_update.html')
    cluster.process()



if __name__ == "__main__":
    sys.exit(main())

