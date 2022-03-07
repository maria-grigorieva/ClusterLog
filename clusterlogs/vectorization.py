import math
import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from .pipeline import Chain


class Vector(Chain):

    def __init__(self, tokenized, w2v_size, w2v_window, cpu_number, model_name):
        self.word2vec = None
        self.tokenized = tokenized
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.cpu_number = cpu_number
        self.model_name = model_name

    def create_word2vec_model(self, min_count=1, iterations=30):
        """
        Train new word2vec model
        :param iterations:
        :param min_count: minimum frequency count of words (recommended value is 10)
        """
        self.word2vec = Word2Vec(self.tokenized,
                                 vector_size=self.w2v_size,
                                 window=self.w2v_window,
                                 min_count=min_count,
                                 workers=self.cpu_number,
                                 epochs=iterations)

        self.word2vec.save(self.model_name)

    def update_word2vec_model(self):
        """
        Retrain word2vec model, taken from file
        """
        try:
            self.word2vec = Word2Vec.load(self.model_name)
        except Exception:
            self.create_word2vec_model()

        self.word2vec.build_vocab(self.tokenized, update=True)
        self.word2vec.train(self.tokenized, total_examples=self.word2vec.corpus_count, epochs=30, report_delay=1)
        self.word2vec.save(self.model_name)

    def load_word2vec_model(self):
        """
        Load word2vec model from file
        """
        self.word2vec = Word2Vec.load(self.model_name)

    def get_w2v_vocabulary(self):
        """
        Returns the vocabulary with word frequencies
        """
        w2c = dict()
        for item in self.word2vec.wv.vocab:
            w2c[item] = self.word2vec.wv.vocab[item].count
        return w2c

    def vectorize_messages(self, tf_idf=False):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        """
        sent2vec = []
        if tf_idf:
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(self.tokenized)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            self.sent2vec = np.array([
                np.mean([self.word2vec[w] * word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.w2v_size)], axis=0)
                for words in self.tokenized
            ])
        else:
            for sent in self.tokenized:
                # sent2vec.append(np.average(self.word2vec[sent],0))
                sent_vec = np.average([self.word2vec.wv.get_vector(w) if w in self.word2vec.wv else np.zeros((self.w2v_size,), dtype=np.float32)
                                       for w in sent], 0)
                sent2vec.append(np.zeros((self.w2v_size,), dtype=np.float32) if np.isnan(np.sum(sent_vec)) else sent_vec)
            self.sent2vec = np.array(sent2vec)

    @staticmethod
    def detect_embedding_size(vocab):
        """
        Automatic detection of word2vec embedding vector size,
        based on the length of vocabulary.
        Max embedding size = 300
        """
        print('Vocabulary size = {}'.format(len(vocab)))
        embedding_size = round(math.sqrt(len(vocab)))
        if embedding_size >= 300:
            embedding_size = 300
        return embedding_size
