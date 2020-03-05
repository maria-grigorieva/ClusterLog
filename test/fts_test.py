import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/fts_10K.csv')
    target = 'message'
    cluster = pipeline.Chain(df.head(500), target, model_name='../models/fts.model', mode='process',
                             add_placeholder=True)
    cluster.process()

    pprint.pprint(cluster.timings)
    pprint.pprint(cluster.result)

    clusters, outliers = cluster.split_clusters(cluster.result, 'cluster_size', 1000)
    pprint.pprint(clusters['pattern'].values)


if __name__ == "__main__":
    sys.exit(main())