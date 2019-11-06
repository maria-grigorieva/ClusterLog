# ClusterLog: Unsupervised Clusterization of Error Messages

**Requirements:**
```
Python >= 3.6 < 3.8
```
This package doesn't work currently with python 2.7 because of `kneed` library, and with python 3.8 because of `gensim`.

```
fuzzywuzzy==0.17.0
gensim==3.4.0
kneed==0.4.1
nltk==3.4.5
numpy==1.16.4
pandas==0.25.1
pyonmttok==1.10.1
scikit-learn==0.21.2
```

**Input:**
   Pandas DataFrame with error log messages. DataFrame may have arbitrary columns and column names, but
   it must contain index column with IDs and column with text log messages. The name of log column is not
   fixed, but it must be specified explicitly in settings as 'target'.
   Possible structure of DataFrame is the following (in this example, `tagret='log_message'`):
   ```
   ID   |   log_message                                                            | timestamp
   -----------------------------------------------------------------------------------------------------
    1   |   No events to process: 16000 (skipEvents) >= 2343 (inputEvents of HITS  | 2019-10-01T10:18:49
    2   |   AODtoDAOD got a SIGKILL signal (exit code 137)                         | 2019-10-01T09:01:57
    ...
   ```

**Output:**
The output is available in different views:
   1) `ALL` - DataFrame grouped by cluster numbers
   2) `INDEX` - dictionary of lists of indexes for all clusters
   3) `TARGET` - dictionary of lists of error messages for all clusters
   4) `cluster labels` - array of cluster labels (as output of DBSCAN -> fit_predict()

**Clusterization of error log messages is implemented as a chain of methods:**

1. *data_preparation* - cleaning initial log messages from unnecessary substrings (UUID, line numbers,...)
2. *tokenization* - split each log message into tokens (NLTK|pyonmttok)
3. *tokens_vectorization* - train word2vec model
4. *sentence_vectorization* - convert word2vec to sent2vec model
5. *kneighbors* - calculate k-neighbors
6. *epsilon_search* - search epsilon for the DBSCAN algorithm
7. *dbscan* - execute DBSCAN clusterization, returns cluster labels


**Requirements:**
```
python >= 3.6 < 3.8
```
```
fuzzywuzzy==0.17.0
gensim==3.8.1
kneed==0.5.0
nltk==3.4.5
numpy==1.16.4
pandas==0.25.1
pyonmttok==1.10.1
scikit-learn==0.21.2
```

**Installation:**

```
pip install clusterlogs
```

**Usage:**
```
from clusterlogs import pipeline
```

1) Download Pandas DataFrame
```
    df = pd.read_csv('<DataFrame>.csv', index_col=0)
    df.set_index('<index column>', inplace=True)
```
2) Define parameters:
```
    clustering_parameters = {'tokenizer':'nltk|pyonmttok',
                             'w2v_size': <size of vector for each token (i.e. 100-300)>,
                             'w2v_window': <size of slicing window for NN algorithms (i.e. 5-10)>,
                             'min_samples': <minimum size of cluster, it's better to set it as 1>}
    target = '<target column with error messages>'
    mode = 'ALL|INDEX|TARGET' (see 'Output' for details)
```


3) Initialize Clustering Pipeline
```
cluster = pipeline.ml_clustering(df, target, clustering_parameters)
```

4) Execute Clustering Pipeline
```
cluster.process()
```

5) Output - depends of mode

```
cluster.clustered_output(mode='INDEX'|'ALL'|'TARGET')
```

mode == 'ALL'
```
    { cluster_1: [
            { feature_1: value, ... },
            ...
            { feature_N: value, ... },
      ],
      ...
      cluster_N: [
            { feature_1: value, ... },
            ...
            { feature_N: value, ... },
      ]
    }
```

mode == 'INDEX'
```
    {
        cluster_1: [<list of IDs>],
        ...
        cluster_N: [<list of IDs>],
    }
```

mode == 'TARGET'
```
    {
        cluster_1: [<list of error messages>],
        ...
        cluster_N: [<list of error messages>],
    }
```

Also, output may be returned as a list of cluster labels:
```
cluster.cluster_labels
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
cluster.statistics()
```

Get all elements of a single cluster:
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
Note: process: <time> - total time of a pipeline
