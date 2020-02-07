import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/all_logs.csv',index_col=0)
    df.dropna(inplace=True)
    target = '0'
    print(df.head())
    cluster = pipeline.ml_clustering(df.head(50000), target, mode='process', model_name='all_logs.model').process()
    #df = pd.read_csv('../samples/harvester_errors24.csv', delimiter=';', index_col=0)
    #df = pd.read_csv('../samples/harvester_errors24.csv', delimiter=';', index_col=0)
    # df = pd.read_csv('../samples/fts_mess_panda.csv', index_col=0)
    #df.set_index('pandaid', inplace=True)
    # To specify clustering parameters, please use dictionary:
    # clustering_parameters = {'tokenizer':'nltk',
    #                          'w2v_size': 300,
    #                          'w2v_window': 10,
    #                          'min_samples': 1}
    #target = 'message'
    #cluster = pipeline.ml_clustering(df, target)
    #cluster.process()

    pprint.pprint(cluster.timings)
    # pprint.pprint(cluster.groups['pattern'].values)
    pprint.pprint(cluster.result)

    pprint.pprint(cluster.postprocessing())
    pprint.pprint(cluster.result_pp)

    stat = cluster.validation(cluster.result)
    pprint.pprint(stat)

    # cluster.split_clusters(stat, 'cluster_size')
    #
    # pprint.pprint(cluster.clusters.shape)
    # pprint.pprint(cluster.outliers.shape)
    #pprint.pprint(cluster.in_cluster(cluster.result, 0))

    #pprint.pprint(cluster.clusters[['pattern']].values)


    # pprint.pprint(cluster.in_cluster(0, 2))

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