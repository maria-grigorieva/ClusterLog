import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/exeerror_jobs_24hours.csv')
    df = df[['pandaid','exeerrordiag']]
    df.set_index('pandaid', inplace=True)

    #df = pd.read_csv('/Users/maria/cernbox/LogsClusterization/Harvester/data_sample30days.csv', sep='\t')
    target = 'exeerrordiag'
    cluster = pipeline.Chain(df, target, mode='update', model_name='../models/exeerror_90days.model',matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/panda_update.html')

    # cluster = pipeline.Chain(df, target, mode='create', model_name='../models/exeerror_tmp.model',matching_accuracy=0.8,
    #                          clustering_type='ML')

    # cluster = pipeline.Chain(df, target, mode='process', model_name='../models/exeerror_docs_90days.model',
    #                          matching_accuracy=0.8,
    #                          clustering_type='ML')
    cluster.process()

    pprint.pprint(cluster.result)
    pprint.pprint(cluster.result['pattern'].values)



if __name__ == "__main__":
    sys.exit(main())