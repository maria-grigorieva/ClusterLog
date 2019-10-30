# ClusterLog: Unsupervised Clusterization of Error Messages

**Input:**
   Pandas DataFrame with error log messages

**Output:**
   Dictionary of clusters

**Clusterization of error log messages is implemented as a chain of methods:**

1. data_preparation - cleaning initial log messages from unnecessary substrings (UUID, line numbers,...)
2. tokenization - split each log message into tokens (NLTK|pyonmttok)
3. tokens_vectorization - train word2vec model
4. sentence_vectorization - convert word2vec to sent2vec model
5. tuning_parameters - search epsilon for the DBSCAN algorithm
6. dbscan - execute DBSCAN clusterization, returns cluster labels

**Usage:**

1) Download Pandas DataFrame
```
    df = pd.read_csv('<DataFrame>.csv')
```
2) Define parameters:
```
    clustering_parameters = {'tokenizer':'nltk|pyonmttok',
                             'w2v_size': <size of vector for each token (i.e. 100-300)>,
                             'w2v_window': <size of slicing window for NN algorithms (i.e. 5-10)>,
                             'min_samples': <minimum size of cluster, it's better to set it as 1>}
    target = '<target column with error messages>'
    index = '<index column of DataFrame>'
    mode = 'ALL|INDEX' (see 'Output' for details)
```


3) Initialize Clustering Pipeline
```
cluster = cluster_pipeline.Cluster(df, index, target, mode, clustering_parameters)
```

4) Execute Clustering Pipeline
```
clustered_df = cluster.process()
```

5) Output (clustered_df) depends of mode

mode == 'ALL'
```
    { cluster: [
            { feature: value, ... },
            { feature: value, ... },
            ...
        ],
        ...
    }
```

mode == 'INDEX'
```
    {
        cluster: [<list of IDs>],
        ...
    }
```

**Additionally:**

Clusters Statistics returns DataFrame with statistic for all clusters:
- "cluster_name" - name of a cluster
- "cluster_size" = number of log messages in cluster
- "first_entry" - first log message in cluster
- "mean_length" - average length of log messages in cluster
- "std_length" - standard deviation of length of log messages in cluster
- "mean_similarity" - average similarity of log messages in cluster
(calculated as the levenshtein distances between the 1st and all other log messages)
= "std_similarity" - standard deviation of similarity of log messages in cluster
```
stats = cluster.statistics(clustered_df)
```

Get elements of a single cluster:
```
cluster.in_cluster(<Cluster_No>)
```

Get tokenized error messages
```
cluster.tokenized
```

Get calculated epsilon value
```
cluster.epsilon
```

Get vocabulary
```
cluster.get_vocabulary()
```

Get word2vec model
```
cluster.word2vec
```

Get timings (execution time for each stage of clustering pipeline)
```
cluster.timings
```
Note: process: <time> - it aa total time of a pipeline