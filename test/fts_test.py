import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/fts_10K.csv')

    # with open("../samples/fts_errors_09_10_Nov_message_id.log") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])
    target = 'message'
    cluster = pipeline.Chain(df, target, model_name='../models/fts.model', mode='update',
                             add_placeholder=True, matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/fts_categorized.html')

    # cluster = pipeline.Chain(df, target, model_name='../models/fts_new.model', mode='create',
    #                          add_placeholder=True, matching_accuracy=0.8,
    #                          clustering_type='ML', output_file='../reports/fts_bigdata_create.html')

    cluster.process()
    #
    # pprint.pprint(cluster.timings)
    # pprint.pprint(cluster.result)
    # big, small = cluster.split_clusters(cluster.result, 'cluster_size', 100)
    # big['pattern'].shape


    #clusters, outliers = cluster.split_clusters(cluster.result, 'cluster_size', 1000)
    # pprint.pprint(cluster.result['pattern'].values)


if __name__ == "__main__":
    sys.exit(main())