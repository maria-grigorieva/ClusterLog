import pandas as pd
import sys
import pprint
from clusterlogs import pipeline, cluster_output

def main():
    df = pd.read_csv('samples/harvester_errors24.csv', sep=';')
    target = 'message'
    mode = 'INDEX'
    cluster = pipeline.ml_clustering(df, target, mode='create', model_name='harvester_test.model')
    cluster.process()

    pprint.pprint(cluster.results)

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

