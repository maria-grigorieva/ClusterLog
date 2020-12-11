from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def vectorize_messages(cleaned_mex, model, tf_idf=False):
    """
    Calculates mathematical average of the word vector representations
    of all the words in each sentence
    """
    sent2vec = []
    tokenized= [row.split() for row in cleaned_mex]
    if tf_idf:
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(tokenized)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        sent2vec = np.array([
            np.mean([model.wv[w] * word2weight[w]
                     for w in words if w in model.wv] or
                    [np.zeros(model.vector_size)], axis=0)
            for words in tokenized
        ])
    else:
        for sent in tokenized:
            sent_vec = np.average([model.wv[w] if w in model.wv else np.zeros((model.vector_size,), dtype=np.float32)
                               for w in sent], 0)
            sent2vec.append(np.zeros((model.vector_size,), dtype=np.float32) if np.isnan(np.sum(sent_vec)) else sent_vec)
        sent2vec = np.array(sent2vec)
    return sent2vec

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print delta loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.loss_vec=[]

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.loss_vec.append(loss)
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.loss_vec.append(loss-self.loss_previous_step)

        self.epoch += 1
        self.loss_previous_step = loss

        
