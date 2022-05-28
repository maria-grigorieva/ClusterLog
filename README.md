# ClusterLog: Unsupervised Clusterization of Error Messages

**Requirements:**
```
Python >= 3.7 < 3.8
```
This package doesn't work currently with python 2.7 because of `kneed` library, and with python 3.8 because of `gensim`.

```
editdistance==0.5.3
gensim==3.8.3
kneed==0.7.0
nltk==3.4.5
numpy==1.18.1
pandas==1.0.1
pyonmttok==1.10.1
scikit-learn==0.22.1
matplotlib==3.1.2
hdbscan==0.8.26
python-rake==1.4.5
pytextrank==2.0.3
Jinja2==2.11.2
spacy==2.3.2
rake-nltk==1.0.4
python-rake==1.4.5
dash==1.17.0
dash-bootstrap-components==0.11.1
dash-extensions==0.0.45
pke @ git+https://github.com/boudinfl/pke.git@6abbcd5d29d14f5151b79a20715ae06dbdd5e4b5
```

Execute in command line to download dictionary required for pyTextRank library
```
python -m spacy download en_core_web_sm
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
Required input:
- `df`
- `target`

Optional input:

- clusterization_settings
    - `w2v_size` (default: 300)
    - `w2v_window` (default: 7)
    - `min_samples` (default: 1)
- `model_name` (path to a file with word2vec model)
- `mode` (create(default)|update|process)
- `output_type` (csv|html)
- `output_fname` (path to output report file)
- `add_placeholder` (default: FALSE)
- `dimensionality_reduction` (default: FALSE)
- `threshold` (clustering threshold, default = 5000)
- `matching_accuracy` (sequences similarity threshold, default = 0.8)
- `clustering_type` (dbscan(default)|hdbscan|k-means|optics|hierarchical|similarity)
- `keywords_extraction` (rake_nltk(default)|RAKE|pyTextRank|lda|ngrams|gensim)
- `categorization` (default: FALSE)

**Modes:**
1) `create`
    - Create word2vec model based on large sample of error logs
    - Save it to file ‘word2vec.model’ on server for further usage
2) `process`
    - Load word2vec model from file (without re-training the model)
3) `update`
    - Load word2vec model from file and train (update) this model with new error logs
    - Save updated model in file


**Clusterization of error log messages is implemented as a chain of methods:**

1) `data_preparation` - cleaning initial log messages from all substrings with digits
2) `grouping equals` - group dataframe by equal cleaned messages
2) `tokenization` - split each log message into tokens (`pyonmttok` + retaining spaces)
3) `tokens_vectorization` - train word2vec model
4) `sentence_vectorization` - convert word2vec to sent2vec model
5) `kneighbors` - calculate k-neighbors
6) `epsilon_search` - search epsilon for the DBSCAN algorithm
7) `dbscan` - execute DBSCAN clusterization, returns cluster labels
8) `reclusterization` - reclustering the existing clusters using the Levenshtein distances between sequences of tokens
9) `validation` - calculating similarity score for each cluster

**Output:**

`Cluster Size` - the number of messages in a cluster
`Patterns` - common textual patterns of messages in a cluster
`Key Phrases` - keywords and key phrases of messages in a cluster

**Clusters statistics:**

Clusters Statistics returns DataFrame or dictionary with statistic for all clusters:
- `cluster_name` - name of a cluster
- `cluster_size` - number of log messages in cluster
- `pattern` - all common substrings in messages in the cluster
- `mean_similarity` - average similarity of log messages in cluster
- `std_similarity` - standard deviation of similarity of log messages in cluster
- `indices` - indices of the initial dataframe, corresponding to the cluster


**Installation:**

```
pip install clusterlogs
```

**Usage:**
```
from clusterlogs.pipeline import Chain

df = pd.read_csv(<CSV file with data>)
target = '<target column with error messages>>'

# Similarity clustering
cluster_sim = Chain(df, target, matching_accuracy=0.8, 
                    clustering_type='similarity', output_type='html',
                    output_fname='<path to output report>', 
                    keywords_extraction='rake_nltk')
cluster_sim.process()

# dbscan (mode=process)
cluster_dbscan_p = Chain(df, target, mode='process', model_name='<path to pre-trained word2vec model>',
                         matching_accuracy=0.5, clustering_type='dbscan', output_type='html',
                         output_fname='<path to output report>')
cluster_dbscan_p.process()

# dbscan (mode=create)
cluster_dbscan_c = Chain(df, target, mode='create', model_name='<path to new word2vec model>',
                         matching_accuracy=0.8, clustering_type='dbscan', output_type='html',
                         output_fname='<path to output report>')
cluster_dbscan_c.process()
```

**Running in MPI-mode**

ClusterLogs supports distributed memory computations at preprocessing and clustering steps with MPI-mode. 

See usage example in `test/pdsdbscan_panda_test.py`. 

Run as: `$ USE_MPI=1 mpirun -n <num_proc> python3 test/pdsdbscan_panda_test.py`

**Author:**
maria.grigorieva@cern.ch (Maria Grigorieva)
