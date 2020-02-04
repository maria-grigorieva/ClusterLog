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

    stat = cluster.validation()
    pprint.pprint(cluster.stat)

    cluster.split_clusters(cluster.stat, 'cluster_size')

    pprint.pprint(cluster.clusters.shape)
    pprint.pprint(cluster.outliers.shape)



if __name__ == "__main__":
    sys.exit(main())