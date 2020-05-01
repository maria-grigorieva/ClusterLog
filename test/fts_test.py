import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/fts_10K.csv')
    target = 'message'
    cluster = pipeline.Chain(df, target, model_name='../models/fts.model', mode='update',
                             add_placeholder=True, matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/fts_update.html')
    cluster.process()
    #
    # pprint.pprint(cluster.timings)
    pprint.pprint(cluster.result)


    #clusters, outliers = cluster.split_clusters(cluster.result, 'cluster_size', 1000)
    pprint.pprint(cluster.result['pattern'].values)


if __name__ == "__main__":
    sys.exit(main())