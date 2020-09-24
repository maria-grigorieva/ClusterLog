import pandas as pd
import sys
sys.path.append('..')
from clusterlogs.pipeline import Chain

def main():
    df = pd.read_csv('../samples/exeerror_1week.csv')
    df = df[['pandaid','exeerrordiag']]
    df.set_index('pandaid', inplace=True)
    target = 'exeerrordiag'

    # similarity
    cluster_sim = Chain(df, target, mode='process', model_name='../models/exeerrors_01-01-20_05-20-20.model',
                             matching_accuracy=0.8, clustering_type='similarity', output_type='html',
                             output_fname='../reports/test_sim', keywords_extraction='rake_nltk')
    cluster_sim.process()
    # dbscan (mode=process)
    cluster_dbscan_p = Chain(df, target, mode='process', model_name='../models/exeerrors_01-01-20_05-20-20.model',
                             matching_accuracy=0.8, clustering_type='dbscan', output_type='html',
                             output_fname='../reports/test_dbscan_p', keywords_extraction='rake_nltk')
    cluster_dbscan_p.process()
    # dbscan (mode=create)
    cluster_dbscan_c = Chain(df, target, mode='create', model_name='../models/exeerrors_tmp.model',
                             matching_accuracy=0.8, clustering_type='dbscan', output_type='html',
                             output_fname='../reports/test_dbscan_c')
    cluster_dbscan_c.process()

if __name__ == "__main__":
    sys.exit(main())