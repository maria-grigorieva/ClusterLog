import editdistance
import numpy as np
from typing import Sequence, Iterable, Hashable, List, Optional, Union
from mpi4py import MPI
import math
import sys

T = Iterable[Hashable]

part_sizes = []


def levenshtein_similarity(a: Sequence[T], b: Sequence[T]) -> float:
    return 1 - editdistance.eval(a, b) / max(len(a), len(b))


def levenshtein_similarity_1_to_n(many: Sequence[Sequence[T]], single: Optional[Sequence[T]] = None) -> Union[List[float], float]:
    if len(many) == 0:
        return 1.
    if single is None:
        single, many = many[0], many[1:]
    if len(many) == 0:
        return [1.0]
    return [levenshtein_similarity(single, item) for item in many]

def parallel_file_read(comm, file_name):
    import os

    if comm != None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        print("file name: {}".format(file_name))
        file_size = os.path.getsize(file_name)
        file_chunk = math.ceil(file_size / comm_size)
        f = open (file_name, "rb")
        if comm_rank != 0:
            f.seek(file_chunk * comm_rank - 1)
            if f.read(1) != b"\n":
                f.readline()
            else:
                print("not \\n")

        start_position = f.tell()
        f.close()
        f = open (file_name, "r")
        f.seek(start_position)
        portion = f.readlines(file_chunk)
    else:
        f = open(file_name, "r")
        portion = f.readlines()

    f.close()

    return portion


def gather_df(comm, df):
    if comm != None:
        import math
        import pandas as pd

        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        if comm_rank == 0:
            blocks = math.ceil(len(df) / 20000)
        else:
            blocks = None
        blocks = comm.bcast(blocks, root=0)
        result = []
        n = 0
        for i in range(blocks):
            n += 1
            start = i * 20000
            if start < len(df):
                end = (i + 1) * 20000
                if end > len(df):
                    end = len(df)
                part = df[start:end]
            else:
                part = None #pd.DataFrame()
            data = comm.gather(part, root=0)
            if comm_rank == 0:
                result.append(data)
        if comm_rank == 0:
            tmp = []
            for d in result:
                tmp.append(pd.concat(d))
            return pd.concat(tmp)
        else:
            return None
    else:
        return df


def split_vectors(comm, vectors):
    """
    Distribute vectorized messages across processes before clustering.
    """
    if comm is None:
        return vectors
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    if comm_size == 1:
        return vectors

    counts = []
    displacements = []

    if rank == 0:
        dims = vectors.shape[1]
        send_buf = np.ravel(vectors)
        default_count = (len(vectors) // comm_size) * dims
        last_pos = 0
        for rank_num in range(comm_size):
            count = default_count
            if rank_num == 0:
                count += (len(vectors) % comm_size) * dims
            counts.append(count)
            displacements.append(last_pos)
            last_pos += count
    else:
        dims = None
        send_buf = None

    dtype = np.float32
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)
    dims = comm.bcast(dims, root=0)

    result = np.empty((counts[rank] // dims, dims), dtype)
    sys.stdout.flush()
    comm.Scatterv([send_buf, counts, displacements, MPI.FLOAT], result, root=0)

    return result


def gather_cluster_labels(comm, labels):
    """
    Gather cluster labels to root process after clustering.
    """
    if comm is None:
        return labels

    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size == 1:
        return labels

    counts = []
    displacements = []
    dtype = labels.dtype
    common_len = comm.allreduce(len(labels))

    default_count = common_len // comm_size
    last_pos = 0
    for rank_num in range(comm_size):
        count = default_count
        if rank_num == 0:
            count += common_len % comm_size
        counts.append(count)
        displacements.append(last_pos)
        last_pos += count

    if rank == 0:
        result = np.empty(common_len, dtype)
    else:
        result = None

    comm.Gatherv(labels, [result, counts, displacements, MPI.INT], root=0)

    return result
