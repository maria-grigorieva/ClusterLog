import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    # df = pd.read_csv('../samples/fts_mess_panda.csv', index_col=0)
    df = pd.read_csv('../samples/error_messages.csv', index_col=0)
    df.set_index('pandaid', inplace=True)
    target = 'exeerrordiag'
    cluster = pipeline.exec(df, target)
    cluster.process()

    pprint.pprint(cluster.timings)
    pprint.pprint(cluster.clusters)
    #pprint.pprint(cluster.clusters[['pattern','cluster_size','mean_similarity']].to_dict('records'))


if __name__ == "__main__":
    sys.exit(main())