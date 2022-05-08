import numpy as np
from mpi4py import MPI
import sys
sys.path.append('..')
from clusterlogs.utility import split_vectors


# run as $ mpirun -n 3 --oversubscribe python3 split_test.py
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    assert comm_size == 3

    to_split = None
    if rank == 0:
        to_split = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ], dtype=np.float32)
    splitted = split_vectors(comm, to_split)
    if rank == 0:
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    elif rank == 1:
        expected = np.array([[7, 8, 9]], dtype=np.float32)
    else:
        expected = np.array([[10, 11, 12]], dtype=np.float32)
    np.testing.assert_array_equal(splitted, expected)

    print('OK!')
