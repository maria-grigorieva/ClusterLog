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

    stat = cluster.validation(cluster.result)
    pprint.pprint(stat)

    clusters, outliers = cluster.split_clusters(stat, 'cluster_size')

    pprint.pprint(clusters.shape)
    pprint.pprint(outliers.shape)



if __name__ == "__main__":
    sys.exit(main())