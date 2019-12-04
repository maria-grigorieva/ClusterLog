import multiprocessing
from math import sqrt
from re import sub
from time import time

import numpy as np
import pandas as pd
import editdistance
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .tokenization import Tokens


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


CLUSTERING_DEFAULTS = {"tokenizer": "nltk",
                       "w2v_size": "auto",
                       "w2v_window": 7,
                       "min_samples": 1}

STATISTICS = ["cluster_name", "cluster_size", "stems", "vocab", "vocab_length",
              "mean_length", "mean_similarity", "std_length", "std_similarity"]


class ml_clustering:


    def __init__(self, df, target, cluster_settings=None, model_name='word2vec.model', mode='create'):
        self.df = df
        self.target = target
        self.set_cluster_settings(cluster_settings or CLUSTERING_DEFAULTS)
        self.cpu_number = self.get_cpu_number()
        self.messages = self.extract_messages()
        self.timings = {}
        self.messages_cleaned = None
        self.tokenized = None
        self.sent2vec = None
        self.distances = None
        self.epsilon = None
        self.cluster_labels = None
        self.model_name = model_name
        self.mode = mode


    @staticmethod
    def get_cpu_number():
        return multiprocessing.cpu_count()


    def set_cluster_settings(self, params):
        for key,value in CLUSTERING_DEFAULTS.items():
            if params.get(key) is not None:
                setattr(self, key, params.get(key))
            else:
                setattr(self, key, value)


    def extract_messages(self):
        """
        Returns a list of all error messages from target column
        :return:
        """
        return list(self.df[self.target])


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
            .dbscan()


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
        :return:
        """
        tokens = Tokens(self.messages, type=self.tokenizer)
        self.tokenized = tokens.process()
        if self.w2v_size == 'auto':
            self.w2v_size = self.detect_embedding_size(tokens)
        return self


    def detect_embedding_size(self, tokens):
        vocab = tokens.get_vocabulary()
        embedding_size = round(len(vocab) ** (2/3))
        if embedding_size >= 400:
            embedding_size = 400
        return embedding_size


    @safe_run
    def tokens_vectorization(self):
        """
        Training word2vec model
        :param iterations:
        :param min_count: minimium frequency count of words (recommended value is 1)
        :return:
        """
        from .vectorization import Vector
        self.word_vector = Vector(self.tokenized,
                                  self.w2v_size,
                                  self.w2v_window,
                                  self.cpu_number,
                                  self.model_name)
        if self.mode == 'create':
            self.word_vector.create_word2vec_model(min_count=1, iterations=10)
        if self.mode == 'update':
            self.word_vector.update_word2vec_model()
        if self.mode == 'process':
            self.word_vector.load_word2vec_model()
        return self


    @safe_run
    def sentence_vectorization(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        self.sent2vec = self.word_vector.sent2vec()
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
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.sent2vec)
        return self


    def clustered_output(self, mode='INDEX'):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        groups = {}
        self.df['cluster'] = self.cluster_labels
        for key, value in self.df.groupby(['cluster']):
            if mode == 'ALL':
                groups[str(key)] = value.to_dict(orient='records')
            elif mode == 'INDEX':
                groups[str(key)] = value.index.values.tolist()
            elif mode == 'TARGET':
                groups[str(key)] = value[self.target].values.tolist()
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
        return ([100 - (editdistance.eval(rows[0], rows[i])*100) / len(rows[0]) for i in range(0, len(rows))])


    def statistics(self, output_mode='frame'):
        """
        Returns dictionary with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "stems" - Longest Common Substring in an Array of Strings
        "vocab" - vocabulary of all messages within the cluster (without punctuation and stop words)
        "vocab_length" - the length of vocabulary
        "mean_length" - average length of log messages in cluster
        "std_length" - standard deviation of length of log messages in cluster
        "mean_similarity" - average similarity of log messages in cluster
        (calculated as the levenshtein distances between the 1st and all other log messages)
        "std_similarity" - standard deviation of similarity of log messages in cluster
        :param clustered_df:
        :param output_mode: frame | dict
        :return:
        """
        clusters = []
        clustered_df = self.clustered_output(mode='TARGET')
        for item in clustered_df:
            row = clustered_df[item]
            stems = row[0] if len(row) == 1 else self.findstem(row)
            lengths = [len(s) for s in row]
            similarity = self.levenshtein_similarity(row)
            tokens = Tokens(row, self.tokenizer)
            tokens.process()
            tokens.clean_tokens()
            vocab = tokens.get_vocabulary()
            vocab_length = len(vocab)
            clusters.append([item,
                             len(row),
                             stems,
                             vocab,
                             vocab_length,
                             np.mean(lengths),
                             np.mean(similarity),
                             np.std(lengths) if len(row)>1 else 0,
                             np.std(similarity) if len(row)>1 else 0])
        df = pd.DataFrame(clusters, columns=STATISTICS).round(2).sort_values(by='cluster_size', ascending=False)
        if output_mode == 'frame':
            return df
        else:
            return df.to_dict(orient='records')

    @staticmethod
    def findstem(arr):
        """
        Find the stem of given list of words
        function to find the stem (longest common substring) from the string array
        :param arr:
        :return:
        """

        # Determine size of the array
        n = len(arr)

        # Take first word from array
        # as reference
        s = arr[0]
        l = len(s)

        res = ""

        for i in range(l):
            for j in range(i + 1, l + 1):

                # generating all possible substrings
                # of our reference string arr[0] i.e s
                stem = s[i:j]
                k = 1
                for k in range(1, n):

                    # Check if the generated stem is
                    # common to all words
                    if stem not in arr[k]:
                        break

                # If current substring is present in
                # all strings and its length is greater
                # than current result
                if (k + 1 == n and len(res) < len(stem)):
                    res = stem

        return res

    @staticmethod
    def distance_curve(distances, mode='show'):
        """
        Save distance curve with knee candidates in file.
        :param distances:
        :param mode: show | save
        :return:
        """
        sensitivity = [1, 3, 5, 10, 100, 150]
        knees = []
        y = list(range(len(distances)))
        for s in sensitivity:
            kl = KneeLocator(distances, y, S=s)
            knees.append(kl.knee)

        plt.style.use('ggplot');
        plt.figure(figsize=(10, 10))
        plt.plot(distances, y)
        colors = ['r', 'g', 'k', 'm', 'c', 'b', 'y']
        for k, c, s in zip(knees, colors, sensitivity):
            plt.vlines(k, 0, len(distances), linestyles='--', colors=c, label=f'S = {s}')
        plt.legend()
        if mode == 'show':
            plt.show()
        else:
            plt.savefig("distance_curve.png")


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
