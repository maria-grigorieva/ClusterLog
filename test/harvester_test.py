import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/harvester_errors24.csv', sep=';')
    #df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/data_sample30days.csv', sep='\t')
    target = 'message'
    cluster = pipeline.Chain(df, target, mode='create', model_name='../models/harvester.model',matching_accuracy=0.8,
                             clustering_type='ML')
    cluster.process()

    pprint.pprint(cluster.result)
    pprint.pprint(cluster.result['pattern'].values)


if __name__ == "__main__":
    sys.exit(main())

