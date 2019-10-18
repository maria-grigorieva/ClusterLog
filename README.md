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