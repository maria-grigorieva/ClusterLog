import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('test_data.csv', index_col=0)
    df.set_index('pandaid', inplace=True)
    clustering_parameters = {'tokenizer':'nltk',
                             'w2v_size': 200,
                             'w2v_window': 5,
                             'min_samples': 1}
    target = 'exeerrordiag'
    mode = 'INDEX'
    cluster = pipeline.ml_clustering(df, target, clustering_parameters)
    cluster.process()
    output = cluster.clustered_output(mode)
    stats = cluster.statistics()

    pprint.pprint(cluster.cluster_labels)
    pprint.pprint(output)
    pprint.pprint(stats)

    pprint.pprint(cluster.in_cluster(1))

    pprint.pprint(cluster.messages_cleaned)
    pprint.pprint(cluster.tokenized)
    pprint.pprint(cluster.epsilon)
    pprint.pprint(cluster.get_vocabulary())
    pprint.pprint(cluster.timings)

if __name__ == "__main__":
    sys.exit(main())