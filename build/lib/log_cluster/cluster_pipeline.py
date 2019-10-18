import numpy as np
from kneed import KneeLocator
import nltk
from nltk.tokenize import TreebankWordTokenizer
nltk.download('words')
nltk.download('stopwords')
from gensim.models import Word2Vec
import math
import pyonmttok
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import re
import logging
import multiprocessing
from statistics import mean, stdev
from fuzzywuzzy import fuzz
import pandas as pd

def safe_run(func):

    def func_wrapper(*args, **kwargs):

        try:
           return func(*args, **kwargs)

        except Exception as e:

            print(e)
            return None

    return func_wrapper

class Cluster:

    def __init__(self, data, mode, clustering_parameters, query_parameters):
        self.data = data
        self.mode = mode
        self.target = query_parameters['target']
        self.index = query_parameters['index']
        self.errors = list(data[self.target])
        self.tokenizer = clustering_parameters['tokenizer']
        self.w2v_size = clustering_parameters['w2v_size']
        self.w2v_window = clustering_parameters['w2v_window']
        self.min_samples = clustering_parameters['min_samples']
        self.cpu_number = multiprocessing.cpu_count()
        self.tokenized = self.data_preparation()
        self.word2vec = self.tokens_vectorization()
        self.sent2vec = self.sentence_vectorization()
        self.tuning_parameters()
        self.cluster_labels = self.dbscan()
        self.clustered_df = self.clustered()

    @safe_run
    def data_preparation(self):
        self.clear_strings()
        return self.tokenization()

    @safe_run
    def tuning_parameters(self):
        self.distances = self.kneighbors()
        #self.distance_curve(self.distances)
        self.epsilon = self.epsilon_search()

    @staticmethod
    def remove_whitespaces(sentence):
        """
        Some error messages has multiple spaces, so we change it to one space.
        :param sentence:
        :return:
        """
        return " ".join(sentence.split())

    @safe_run
    def clear_strings(self):
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

        for idx, item in enumerate(self.errors):
            _cleaned = re.sub(_line_number, "at line LINE_NUMBER", item)
            _cleaned = re.sub(_uid, "UID", _cleaned)
            _cleaned = re.sub(_uuid, "UUID", _cleaned)
            self.errors[idx] = self.remove_whitespaces(_cleaned)

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
            for line in self.errors:
                tokenized.append(TreebankWordTokenizer().tokenize(line))
            logging.info("Stage 1 finished")
        elif self.tokenizer  == 'pyonmttok':
            tokenizer = pyonmttok.Tokenizer("space", joiner_annotate=False, segment_numbers=False)
            for doc in self.errors:
                tokens, features = tokenizer.tokenize(doc)
                tokenized.append(tokens)
        return tokenized

    @safe_run
    def tokens_vectorization(self, min_count=1, iter=10):
        """
        Training word2vec model
        :param min_count: minimium frequency count of words (recommended value is 1)
        :param iter: (recommended value is 10)
        :return:
        """
        word2vec = Word2Vec(self.tokenized, size=self.w2v_size, window=self.w2v_window,
                            min_count=min_count, workers=self.cpu_number, iter=iter)
        return word2vec

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
                    if numw == 0:
                        sent_vec = self.word2vec[w]
                    else:
                        sent_vec = np.add(sent_vec, self.word2vec[w])
                    numw += 1
                except Exception as error:
                    pass

            sent2vec.append(np.asarray(sent_vec) / numw)
        return np.array(sent2vec)

    @safe_run
    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(math.sqrt(len(self.sent2vec)))
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(self.sent2vec)
        distances, indices = nbrs.kneighbors(self.sent2vec)
        distances = np.sort(distances, axis=0)
        if k > 2:
            avg_distances = []
            for line in distances:
                avg_distances.append(mean(line))
            return avg_distances
        else:
            return distances[:, 1]

    @safe_run
    def epsilon_search(self):
        """
        Search epsilon for DBSCAN
        :return:
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))))
        if len(kneedle.all_elbows)>0:
            return max(kneedle.all_elbows)
        else:
            return 1

    @safe_run
    def dbscan(self):
        """
        DBSCAN clusteing with daal4py library
        :return: DBSCAN labels
        """
        return DBSCAN(eps=self.epsilon, min_samples=self.min_samples, n_jobs=self.cpu_number).fit_predict(self.sent2vec)
        # algo = daal4py.dbscan(minObservations=self.min_samples, epsilon=self.epsilon,
        #                       resultsToCompute='computeCoreIndices|computeCoreObservations')
        # result = algo.compute(self.sent2vec)
        # return result.assignments[:, 0].tolist()


    @safe_run
    def clustered(self):
        """
        Returns dictionary of clusters with the arrays of elements
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

    @safe_run
    def errors_in_cluster(self, cluster_label):
        results = []
        for idx, l in enumerate(self.cluster_labels):
            if l == cluster_label:
                results.append(self.errors[idx])
        return results

    @safe_run
    def cluster_statistics(self):
        clusters = []
        for item in self.clustered_errors:
            cluster = {}
            cluster["cluster_name"] = item
            cluster["first_entry"] = self.clustered_errors[item][0]['error_message']
            cluster["cluster_size"] = len(self.clustered_errors[item])
            lengths = []
            for s in self.clustered_errors[item]:
                lengths.append(len(s['error_message']))
            mean_length = mean(lengths)
            try:
                std_length = stdev(lengths)
            except Exception as error:
                std_length = 0
            cluster["mean_length"] = mean_length
            cluster["std_lengt"] = std_length
            x0 = self.clustered_errors[item][0]['error_message']
            dist = []
            for i in range(0, len(self.clustered_errors[item])):
                x = self.clustered_errors[item][i]['error_message']
                dist.append(fuzz.ratio(x0, x))
            cluster["mean_similarity"] = mean(dist)
            try:
                cluster["std_similarity"] = stdev(dist)
            except Exception as error:
                cluster["std_similarity"] = 0
            clusters.append(cluster)
        clusters_df = pd.DataFrame(clusters).round(2)
        return clusters_df.T.to_dict()