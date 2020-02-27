import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/harvester_errors24.csv', sep=';')
    #df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/data_sample30days.csv', sep='\t')
    target = 'message'
    mode = 'INDEX'
    cluster = pipeline.Chain(df.head(1000), target, mode='create', model_name='../models/harvester.model',matching_accuracy=0.6)
    cluster.process()

    pprint.pprint(cluster.result)

    # output = cluster.clustered_output(mode)
    # stats = cluster.statistics(output_mode='dict')

    # pprint.pprint(cluster.cluster_labels)
    # pprint.pprint(output)
    # pprint.pprint(stats)
    #

    #
    # pprint.pprint(cluster.messages_cleaned)
    # pprint.pprint(cluster.tokenized)
    # pprint.pprint(cluster.epsilon)
    # pprint.pprint(cluster.timings)
    # cluster.distance_curve(cluster.distances, 'save')

if __name__ == "__main__":
    sys.exit(main())

