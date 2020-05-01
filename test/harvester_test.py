import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    #df = pd.read_csv('../samples/Harvester/harv_1day.csv', sep='\t')
    df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/harvester_errors24.csv', sep=';')
    target = 'message'
    cluster = pipeline.Chain(df, target, mode='update', model_name='../models/harvester_30days.model',matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/harvester24_update.html')
    cluster.process()

    pprint.pprint(cluster.result)
    pprint.pprint(cluster.result['pattern'].values)


if __name__ == "__main__":
    sys.exit(main())

