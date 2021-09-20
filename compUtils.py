import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType
from pyspark.sql.functions import col

def readCompareDist(file1, file2):
    df_1=pd.read_csv('{}_clusters.csv'.format(file1))
    df_2=pd.read_csv('{}_clusters.csv'.format(file2))
    f,ax=plt.subplots(figsize=(20,8))
    bins=np.arange(0,160000,800)
    ax.hist(df_1['cluster_size'].values,bins=bins,alpha=0.9,label=file1)
    ax.hist(df_2['cluster_size'].values,bins=bins,alpha=0.5,label=file2)

    ax.set_xlabel('cluster_size')
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_yscale('log')
    ax.set_title("Distribution of clusters")
    ax.legend(['{}: {} clusters'.format(file1,df_1.shape[0]),'{}: {} clusters'.format(file2,df_2.shape[0])])
    plt.show()
    f.savefig("{}VS{}.pdf".format(file1,file2))
    return df_1,df_2

def compareDist(df_1, df_2, label1, label2,bins):    
    f,ax=plt.subplots(figsize=(20,8))
    bins=bins
    ax.hist(df_1['cluster_size'].values,bins=bins,alpha=0.9,label=label1)
    ax.hist(df_2['cluster_size'].values,bins=bins,alpha=0.5,label=label2)

    ax.set_xlabel('cluster_size')
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_yscale('log')
    ax.set_title("Distribution of clusters")
    ax.legend(['{}: {} clusters'.format(label1,df_1.shape[0]),'{}: {} clusters'.format(label2,df_2.shape[0])])
    plt.show()
    f.savefig("{}VS{}.pdf".format(label1,label2))
   

def readData(date,spark):
    _schema = StructType([
    StructField('metadata', StructType([StructField('timestamp',LongType(), nullable=True)])),
    StructField('data', StructType([
        StructField('endpnt', StringType(), nullable=True),
        StructField('src_hostname', StringType(), nullable=True),
        StructField('dst_hostname', StringType(), nullable=True),
        StructField('t_error_code', IntegerType(), nullable=True),
        StructField('tr_error_category', StringType(), nullable=True),
        StructField('tr_error_scope', StringType(), nullable=True),
        StructField('t_failure_phase', StringType(), nullable=True),
        StructField('t__error_message', StringType(), nullable=True)])),])
    fts_df = spark.read.json('/project/monitoring/archive/fts/raw/complete/{}'.format(date),schema=_schema)
    fts_df = fts_df.select(
    col('metadata.timestamp').alias('time'),
    col('data.src_hostname').alias('src'),
    col('data.endpnt').alias('endpnt'),
    col('data.dst_hostname').alias('dst'),
    col('data.t_error_code').alias('error_code'),
    col('data.tr_error_category').alias('error_category'),
    col('data.tr_error_scope').alias('error_scope'),
    col('data.t_failure_phase').alias('failure_phase'),        
    col('data.t__error_message').alias('error_message')).where('error_message <> ""')
    df = fts_df.toPandas()
    df.drop_duplicates(inplace=True)
    
    return df

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def readDateVector(dateVec,spark):
    _schema = StructType([
    StructField('metadata', StructType([StructField('timestamp',LongType(), nullable=True)])),
    StructField('data', StructType([
        StructField('t__error_message', StringType(), nullable=True),
        StructField('src_hostname', StringType(), nullable=True),
        StructField('dst_hostname', StringType(), nullable=True)])),])
    fts_df = spark.read.json(dateVec,schema=_schema)
    fts_df = fts_df.select(
    col('metadata.timestamp').alias('time'),
    col('data.src_hostname').alias('src'),
    col('data.dst_hostname').alias('dst'),
    col('data.t__error_message').alias('error_message')).where('error_message <> ""')
    df = fts_df.toPandas()
    return df
