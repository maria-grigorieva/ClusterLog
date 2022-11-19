import pandas as pd
import sys
import io
from mpi4py import MPI

sys.path.append('..')
from clusterlogs.pipeline import Chain
from clusterlogs.utility import parallel_file_read


def main():
    comm = MPI.COMM_WORLD
    lines = parallel_file_read(comm, '../samples/fts_10K_no_header.csv')
    header = ['pandaid,message\n']
    lines = header + lines
    raw_input = ''.join(lines)
    df = pd.read_csv(io.StringIO(raw_input))
    df.set_index('pandaid', inplace=True)
    target = 'message'

    cluster_pdsdbscand = Chain(df, target, mode='process', model_name='../models/exeerrors_01-01-20_05-20-20.model',
                               matching_accuracy=0.5, clustering_type='dbscan', output_type='html',
                               output_fname='../reports/test_dbscan_fts', keywords_extraction='rake_nltk')
    cluster_pdsdbscand.timings = {
        'tokenization': 0,
        'tokens_vectorization': 0,
        'sentence_vectorization': 0,
        'ml_clustering': 0,
        'clusters_description': 0,
    }

    cluster_pdsdbscand.process()


if __name__ == "__main__":
    sys.exit(main())
