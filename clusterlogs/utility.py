import editdistance

from math import ceil
from pandas import concat
from os.path import getsize
from typing import Sequence, Iterable, Hashable, List, Optional, Union

T = Iterable[Hashable]


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
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        print("file name: {}".format(file_name))
        file_size = getsize(file_name)
        file_chunk = ceil(file_size / comm_size)
        f = open(file_name, "rb")
        if comm_rank != 0:
            f.seek(file_chunk * comm_rank - 1)
            if f.read(1) != b"\n":
                f.readline()
            else:
                print("not \\n")

        start_position = f.tell()
        f.close()
        f = open(file_name, "r")
        f.seek(start_position)
        portion = f.readlines(file_chunk)
    else:
        f = open(file_name, "r")
        portion = f.readlines()

    f.close()

    return portion


def gather_df(comm, df):
    if comm is None:
        return df

    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        blocks = ceil(len(df) / 20000)
    else:
        blocks = None
    blocks = comm.bcast(blocks, root=0)

    result = []
    # n = 0
    for i in range(blocks):
        # n += 1
        start = i * 20000
        if start < len(df):
            end = (i + 1) * 20000
            if end > len(df):
                end = len(df)
            part = df[start:end]
        else:
            part = None  # pd.DataFrame()
        data = comm.gather(part, root=0)
        if comm_rank == 0:
            result.append(data)
    if comm_rank == 0:
        tmp = []
        for d in result:
            tmp.append(concat(d))
        return concat(tmp)
    else:
        return None
