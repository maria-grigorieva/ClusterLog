import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../fts_10K.csv')
    target = 'message'
    cluster = pipeline.ml_clustering(df, target, model_name='fts.model', mode='process')
    cluster.process()

    pprint.pprint(cluster.timings)
    pprint.pprint(cluster.result)





    # pprint.pprint(cluster.in_cluster(cluster.clusters.result, 80))
    # pprint.pprint(cluster.clusters.result.sort_values(by=['cluster_size'],ascending=False)['pattern'].values)
    #
    #
    #
    # stat = cluster.validation(cluster.result)
    # pprint.pprint(stat)
    #
    # pprint.pprint(stat.describe())
    #
    # clusters, outliers = cluster.split_clusters(stat, 'cluster_size')
    #
    # garbage = cluster.garbage_collector(cluster.result)
    #
    # cluster.postprocessing(cluster.result)
    #
    # pprint.pprint(cluster.result_pp)
    #
    # pprint.pprint(cluster.result_pp.sort_values(by=['cluster_size'],ascending=False)['pattern'].values)
    # #garbage = cluster.garbage_collector(cluster.result)
    #
    # clusters, outliers = cluster.split_clusters(cluster.result_pp, 'cluster_size')
    #
    # pprint.pprint(clusters)
    #
    # stat_pp = cluster.validation(cluster.result_pp)
    # pprint.pprint(stat_pp.describe())



if __name__ == "__main__":
    sys.exit(main())