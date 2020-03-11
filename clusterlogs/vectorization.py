from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from .pipeline import Chain
import math

class Vector(Chain):

    def __init__(self, tokenized, w2v_size, w2v_window, cpu_number, model_name):
        self.word2vec = None
        self.tokenized = tokenized
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.cpu_number = cpu_number
        self.model_name = model_name
        self.doc2vec = None



    def create_doc2vec_model(self):
        tagged_docs = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(self.tokenized)]
        self.doc2vec = Doc2Vec(tagged_docs, vector_size=self.w2v_size, window=self.w2v_window, workers=self.cpu_number)
        return self.doc2vec.docvecs.vectors_docs


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
        try:
            self.word2vec = Word2Vec.load(self.model_name)
        except Exception as e:
            self.create_word2vec_model()

        self.word2vec.build_vocab(self.tokenized, update=True)
        self.word2vec.train(self.tokenized, total_examples=self.word2vec.corpus_count, epochs=30, report_delay=1)
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


    def vectorize_messages(self):
        """
        Calculates mathematical average of the word vector representations
        of all the words in each sentence
        :return:
        """
        sent2vec = []
        for sent in self.tokenized:
            sent_vec = np.average([self.word2vec[w] if w in self.word2vec else np.zeros((self.w2v_size,), dtype=np.float32)
                                   for w in sent], 0)
            sent2vec.append(np.zeros((self.w2v_size,), dtype=np.float32) if np.isnan(np.sum(sent_vec)) else sent_vec)
        self.sent2vec = np.array(sent2vec)


    @staticmethod
    def detect_embedding_size(vocab):
        """
        Automatic detection of word2vec embedding vector size,
        based on the length of vocabulary.
        Max embedding size = 300
        :return:
        """
        print('Vocabulary size = {}'.format(len(vocab)))
        embedding_size = round(math.sqrt(len(vocab)))
        if embedding_size >= 300:
            embedding_size = 300
        return embedding_size
