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
    data_parameters = {'target': 'exeerrordiag',
                       'index': 'pandaid'}
    cluster = cluster_pipeline.Cluster(df, 'ALL', clustering_parameters, data_parameters)
    clustered_df = cluster.process()
    stats = cluster.statistics(clustered_df)

    pprint.pprint(clustered_df)
    pprint.pprint(stats)

    pprint.pprint(cluster.errors_in_cluster(1))

if __name__ == "__main__":
    sys.exit(main())