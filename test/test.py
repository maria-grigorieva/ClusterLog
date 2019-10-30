from log_cluster import cluster_pipeline
import pandas as pd
import sys
import pprint


def main():
    df = pd.read_csv('test_data.csv')
    clustering_parameters = {'tokenizer':'nltk',
                             'w2v_size': 200,
                             'w2v_window': 5,
                             'min_samples': 1}
    target = 'exeerrordiag'
    index = 'pandaid'
    mode = 'ALL'
    cluster = cluster_pipeline.Cluster(df, index, target, mode, clustering_parameters)
    clustered_df = cluster.process()
    stats = cluster.statistics(clustered_df)

    pprint.pprint(clustered_df)
    pprint.pprint(stats)

    pprint.pprint(cluster.in_cluster(1))

    pprint.pprint(cluster.messages_cleaned)
    pprint.pprint(cluster.tokenized)
    pprint.pprint(cluster.epsilon)
    pprint.pprint(cluster.get_vocabulary())
    pprint.pprint(cluster.timings)

if __name__ == "__main__":
    sys.exit(main())