import numpy as np
from mpi4py import MPI
import sys
sys.path.append('..')
from clusterlogs.utility import split_vectors_by_parts, gather_vectors_by_parts


# run as $ mpirun -n 3 --oversubscribe python3 split_gather_by_parts_test.py
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    assert comm_size == 3

    # Test split
    to_split = None
    if rank == 0:
        to_split = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
        ])
    splitted = split_vectors_by_parts(comm, to_split, 4)
    if rank == 0:
        expected = np.array([
            [1, 2, 3], [4, 5, 6],
            [13, 14, 15], [16, 17, 18],
            [25, 26, 27],
        ])
    elif rank == 1:
        expected = np.array([[7, 8, 9], [19, 20, 21], [28, 29, 30]])
    else:
        expected = np.array([[10, 11, 12], [22, 23, 24], [31, 32, 33]])
    np.testing.assert_array_equal(splitted, expected)
    # Test split  ^

    # Test gather
    gathered = gather_vectors_by_parts(comm, splitted)
    if rank == 0:
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
        ])
    else:
        expected = None
    np.testing.assert_array_equal(gathered, expected)
    # Test gather ^

    print('OK!')
