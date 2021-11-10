import editdistance
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
    import math
    import os

    if comm != None and comm.Get_size() > 1:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

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

def parallel_db_read(comm, dbname, user, host='localhost', password=None, date_range=None, limit=None):
    import psycopg2
    conn = psycopg2.connect("dbname={} user={} host={} password='{}'".format(dbname, user, host, password)
    with conn.cursor() as cur:
        query = "INSERT INTO clusterlog_runs (end_time) VALUES (CURRENT_TIMESTAMP) RETURNING run_id;"
        run_id = None
        if comm.Get_rank() == 0:
            print("before create run")
            cur.execute(query)
            run_id = int(cur.fetchone()[0])
            print("after create run {}".format(run_id))
        run_id = comm.bcast(run_id, 0)
        query = "CREATE TABLE temp_messages_{} as SELECT error_messages.pandaid, message FROM panda_raw,  error_messages where panda_raw.pandaid = error_messages.pandaid".format(run_id)
        if date_range != None:
            where_query = " AND creationtime >= '{}' AND creationtime <= '{}'".format(*date_range)
            query = query + where_query
        if limit != None:
            query = query + " LIMIT {}".format(limit)
        query = query + ";"
        lines_num = None
        if comm.Get_rank() == 0:
            print("before copy data")
            try:
                print(query)
                cur.execute(query)
                cur.execute("COMMIT")
            except Exception as err:
                print(err)
                exit()
            print("after copy data")
            #cur.fetchall()
            print("before getting lines_num")
            cur.execute("SELECT count(*) FROM temp_messages_{};".format(run_id))
            lines_num = min(int(cur.fetchone()[0]), limit)
            print("after getting lines_num {}".format(lines_num))
        lines_num = comm.bcast(lines_num, 0)
        if comm.Get_rank() == 0:
            print("starage:db;processes:{};lines:{}".format(comm.Get_size(), lines_num))
        lines_num_p = lines_num / comm.Get_size();
        query = "SELECT pandaid, message FROM temp_messages_{}".format(run_id)
        query = query + " LIMIT {} OFFSET {};".format(lines_num_p, lines_num_p * comm.Get_rank())
        cur.execute(query)
        lines = cur.fetchall()
        comm.barrier()
        if comm.Get_rank() == 0:
            cur.execute("DROP TABLE temp_messages_{};".format(run_id))
    conn.close()

    return lines, run_id

def gather_df(comm, df):
    if comm != None and comm.Get_size() > 1:
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
