import multiprocessing
from math import sqrt
from re import sub
from statistics import mean, stdev
from time import time

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
from kneed import KneeLocator
from nltk.tokenize import TreebankWordTokenizer
from pyonmttok import Tokenizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def safe_run(method):
    def func_wrapper(self, *args, **kwargs):

        try:
            ts = time()
            result = method(self, *args, **kwargs)
            te = time()
            self.timings[method.__name__] = round((te - ts), 4)
            return result

        except Exception:
            return None

    return func_wrapper


CLUSTERING_SETTINGS = ["tokenizer", "w2v_size", "w2v_window", "min_samples"]

STATISTICS = ["cluster_name", "cluster_size", "first_entry",
              "mean_length", "mean_similarity", "std_length", "std_similarity"]


class Cluster:
    def __init__(self, data, index, target, mode, _cluster):
        # Initialize Pandas DataFrame
        self.data = data
        # Index column
        self.index = index
        # Target column for clusterization
        self.target = target
        # ALL | INDEX
        self.mode = mode
        # Initialize clusterization settings
        self.set_cluster_settings(_cluster)
        self.cpu_number = self.get_cpu_number()
        self.messages = self.extract_messages()
        self.timings = {}
        self.messages_cleaned = None
        # Tokenized error messages (a list of tokens for each message)
        self.tokenized = None
        # Word2Vec Model
        self.word2vec = None
        # Sentence2Vec Model
        self.sent2vec = None
        # K-neighbors distances for sent2vec
        self.distances = None
        # Epsilon value for DBSCAN clusterization algorithm
        self.epsilon = None
        self.cluster_labels = None


    @staticmethod
    def get_cpu_number():
        return multiprocessing.cpu_count()

    def set_cluster_settings(self, params):
        for key in CLUSTERING_SETTINGS:
            setattr(self, key, params.get(key))

    def extract_messages(self):
        """
        Returns a list of all error messages from target column
        :return:
        """
        return list(self.data[self.target])

    @safe_run
    def process(self):
        """
        Chain of methods, providing data preparation, vectorization and clusterization
        :return:
        """
        return self.data_preparation() \
            .tokenization() \
            .tokens_vectorization() \
            .sentence_vectorization() \
            .kneighbors() \
            .epsilon_search() \
            .dbscan() \
            .clustered()

    @safe_run
    def data_preparation(self):
        """
        Cleaning log messages from unnucessary substrings and tokenization
        :return:
        """
        self.messages_cleaned = cleaner(self.messages)
        return self


    @safe_run
    def tokenization(self):
        """
        Tokenization of a list of error messages.
        The best tokenizer for error messages is TreebankWordTokenizer (nltk).
        It's good at tokenizing file paths.
        Alternative tokenizer. It performs much faster, but worse in tokenizing of paths.
        It splits all paths by "/".
        TODO: This method should be optimized to the same tokenization quality as TreebankWordTokenizer
        :return:
        """
        tokenized = []
        if self.tokenizer == 'nltk':
            for line in self.messages:
                tokenized.append(TreebankWordTokenizer().tokenize(line))
        elif self.tokenizer == 'pyonmttok':
            tokenizer = Tokenizer("space", joiner_annotate=False, segment_numbers=False)
            for line in self.messages:
                tokens, features = tokenizer.tokenize(line)
                tokenized.append(tokens)
        self.tokenized = tokenized
        return self


    @safe_run
    def tokens_vectorization(self, min_count=1, iterations=10):
        """
        Training word2vec model
        :param iterations:
        :param min_count: minimium frequency count of words (recommended value is 1)
        :return:
        """
        self.word2vec = Word2Vec(self.tokenized,
                                 size=self.w2v_size,
                                 window=self.w2v_window,
                                 min_count=min_count,
                                 workers=self.cpu_number,
                                 iter=iterations)
        return self


    def get_vocabulary(self):
        """
        Returns the vocabulary with word frequencies
        :return:
        """
        w2c = dict()
        for item in self.word2vec.wv.vocab:
            w2c[item] = self.word2vec.wv.vocab[item].count
        return w2c


    @safe_run
    def sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        sent2vec = []
        for sent in self.tokenized:
            sent_vec = []
            numw = 0
            for w in sent:
                try:
                    sent_vec = self.word2vec[w] if numw == 0 else np.add(sent_vec, self.word2vec[w])
                    numw += 1
                except Exception:
                    pass
            sent2vec.append(np.asarray(sent_vec) / numw)
        self.sent2vec = np.array(sent2vec)
        return self


    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(sqrt(len(self.sent2vec)))
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(self.sent2vec)
        distances, indices = nbrs.kneighbors(self.sent2vec)
        self.distances = [np.mean(d) for d in np.sort(distances, axis=0)]
        return self


    @safe_run
    def epsilon_search(self):
        """
        Search epsilon for the DBSCAN clusterization
        :return:
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))))
        self.epsilon = max(kneedle.all_elbows) if (len(kneedle.all_elbows) > 0) else 1
        return self


    @safe_run
    def dbscan(self):
        """
        Execution of the DBSCAN clusterization algorithm.
        Returns cluster labels
        :return:
        """
        self.cluster_labels = DBSCAN(eps=self.epsilon, min_samples=self.min_samples, n_jobs=self.cpu_number) \
            .fit_predict(self.sent2vec)
        return self


    def clustered(self):
        """
        Returns dictionary of clusters with the arrays of elements
        If mode == ALL:
        { "<cluster_1>": [
                "<feature_name_1>": "<value>",
                "<feature_name_2>": "<value>",
                ...
            ],
          "<cluster_2>": [...],
            ...
        }
        If mode == INDEX:
        {
            "<cluster_1>": [id_1, id_2, id_3, ...id_N],
            "<cluster_2>": [id_x, id_y, id_z, ...id_M],
            ...
        }
        :return:
        """
        groups = {}
        self.data['cluster'] = self.cluster_labels
        for key, value in self.data.groupby(['cluster']):
            if self.mode == 'ALL':
                groups[str(key)] = value.to_dict(orient='records')
            elif self.mode == 'INDEX':
                groups[str(key)] = value[self.index].values.tolist()
        return groups


    def in_cluster(self, cluster_label):
        """
        Returns all log messages in particular cluster
        :param cluster_label:
        :return:
        """
        results = []
        for idx, l in enumerate(self.cluster_labels):
            if l == cluster_label:
                results.append(self.messages[idx])
        return results


    def levenshtein_similarity(self, rows):
        """
        Takes a list of log messages and calculates similarity between
        first and all other messages.
        :param rows:
        :return:
        """
        x0 = rows[0][self.target]
        return ([fuzz.ratio(x0, rows[i][self.target]) for i in range(0, len(rows))])


    def statistics(self, clustered_df):
        """
        Returns DataFrame with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "first_entry" - first log message in cluster
        "mean_length" - average length of log messages in cluster
        "std_length" - standard deviation of length of log messages in cluster
        "mean_similarity" - average similarity of log messages in cluster
        (calculated as the levenshtein distances between the 1st and all other log messages)
        "std_similarity" - standard deviation of similarity of log messages in cluster
        :param clustered_df:
        :return:
        """
        clusters = []
        for item in clustered_df:
            rows = clustered_df[item]
            lengths = [len(s[self.target]) for s in rows]
            similarity = self.levenshtein_similarity(rows)
            clusters.append([item,
                             len(rows),
                             rows[0][self.target],
                             mean(lengths),
                             mean(similarity),
                             stdev(lengths) if len(rows)>1 else 0,
                             stdev(similarity) if len(rows)>1 else 0])
        df = pd.DataFrame(clusters, columns=STATISTICS).round(2)
        return df.T.to_dict()


def remove_whitespaces(sentence):
    """
    Some error messages has multiple spaces, so we change it to one space.
    :param sentence:
    :return:
    """
    return " ".join(sentence.split())


def cleaner(messages):
    """
    Clear error messages from unnecessary data:
    - UID/UUID in file paths
    - line numbers - as an example "error at line number ..."
    Removed parts of text are substituted with titles
    :return:
    """
    _uid = r'[0-9a-zA-Z]{12,128}'
    _line_number = r'(at line[:]*\s*\d+)'
    _uuid = r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'

    for idx, item in enumerate(messages):
        _cleaned = sub(_line_number, "at line LINE_NUMBER", item)
        _cleaned = sub(_uid, "UID", _cleaned)
        _cleaned = sub(_uuid, "UUID", _cleaned)
        messages[idx] = remove_whitespaces(_cleaned)
    return messages
