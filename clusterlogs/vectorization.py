from gensim.models import Word2Vec
import numpy as np
from .pipeline import ml_clustering

class Vector(ml_clustering):

    def __init__(self, tokenized, w2v_size, w2v_window, cpu_number, model_name):
        self.word2vec = None
        self.tokenized = tokenized
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.cpu_number = cpu_number
        self.model_name = model_name


    def create_word2vec_model(self, min_count=1, iterations=10):
        """
        Train new word2vec model
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

        self.word2vec.save(self.model_name)


    def update_word2vec_model(self):
        """
        Retrain word2vec model, taken from file
        :return:
        """
        self.word2vec = Word2Vec.load(self.model_name)
        self.word2vec.train(self.tokenized,
                            total_examples=self.word2vec.corpus_count,
                            epochs=30,
                            report_delay=1)
        self.word2vec.save(self.model_name)


    def load_word2vec_model(self):
        """
        Load word2vec model from file
        :return:
        """
        self.word2vec = Word2Vec.load(self.model_name)


    def get_w2v_vocabulary(self):
        """
        Returns the vocabulary with word frequencies
        :return:
        """
        w2c = dict()
        for item in self.word2vec.wv.vocab:
            w2c[item] = self.word2vec.wv.vocab[item].count
        return w2c


    def sent2vec(self):
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
            row = np.zeros((self.w2v_size,), dtype=np.float32) if len(sent_vec) == 0 else np.asarray(sent_vec) / numw
            sent2vec.append(row)
        return np.array(sent2vec)
