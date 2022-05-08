import numpy as np
from mpi4py import MPI
import sys
sys.path.append('..')
from clusterlogs.utility import gather_cluster_labels


# run as $ mpirun -n 3 --oversubscribe python3 gather_test.py
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    assert comm_size == 3

    if rank == 0:
        splitted = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    elif rank == 1:
        splitted = np.array([6, 7, 8], dtype=np.int32)
    else:
        splitted = np.array([9, 10, 11], dtype=np.int32)
    gathered = gather_cluster_labels(comm, splitted)
    if rank == 0:
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        np.testing.assert_array_equal(gathered, expected)
    else:
        assert gathered is None

    print('OK!')
