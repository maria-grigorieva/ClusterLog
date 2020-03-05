import math
import numpy as np
from string import punctuation
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from .tokenization import Tokens
import warnings
import pprint


class TermsAnalysis:

    def __init__(self, tokens):

        self.tokens = tokens


    def process(self):

        print('Initial number of error messages: {}'.format(len(self.tokens.tokenized_pattern)))
        print('Initial size of vocabulary: {}'.format(len(self.tokens.vocabulary_pattern)))
        # get only unique messages
        unique_tokenized = np.unique(self.tokens.tokenized_pattern)
        print('Number of unique messages: {}'.format(len(unique_tokenized)))
        # convert all tokens to lower case
        unique_tokenized = self.tokens.to_lower(unique_tokenized)
        # clean tokens from stop words and punctuation (it's necessary for the TF-IDF analysis)
        unique_tokenized = self.tokens.clean_tokenized(unique_tokenized)
        # get frequence of cleaned tokens
        frequency = Tokens.get_term_frequencies(unique_tokenized)
        # remove tokens that appear only once and save tokens which are textual substrings
        unique_tokenized = [
            [token for token in row if frequency[token] > 1]
            for row in unique_tokenized]
        print('Size of vocabulary after removing stop tokens and tokens, that appear only once: {}'.
              format(len(Tokens.get_vocabulary(unique_tokenized))))
        print(Tokens.get_vocabulary(unique_tokenized))
        # # TF-IDF Terms Analysis
        # dct = Dictionary(unique_tokenized)
        # corpus = [dct.doc2bow(line) for line in unique_tokenized]
        # tfidf = TfidfModel(corpus, normalize=True)
        # corpus_tfidf = tfidf[corpus]
        # d = {dct.get(id): value for doc in corpus_tfidf for id, value in doc}
        # for k,v in d.items():
        #     if k.isalpha():
        #         d[k] = 0.00
        # weights = [x[1] for x in d.items()]
        # max = np.max(weights)
        # print(max)
        # std = np.std(weights)
        # print(std)
        # top = round((max - std),1)
        # #top = round((np.mean(weights) + np.std(weights)),1)
        # top = np.mean(weights)
        # print(top)
        # s = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        # print(list(s.items()))
        #
        # tokenized_tfidf = []
        # # take initial messages and remove all rare tokens
        # # the token is rare if it's weight is more than top
        # for i,row in enumerate(self.tokens.tokenized_pattern):
        #     # print(row)
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore", category=RuntimeWarning)
        #         try:
        #             # tokenized_tfidf.append([token if
        #             #                         (token in d and d[token] < top) or
        #             #                         token.isalpha() or
        #             #                         token in punctuation or
        #             #                         token in u"\u2581"
        #             #                         else '｟*｠' for token in row])
        #             tokenized_tfidf.append([token for token in row if
        #                                     (token.lower() in d and d[token.lower()] < top) or
        #                                     token.lower().isalpha()])
        #         except Exception as e:
        #             print(row)
        #
        # vocab = Tokens.get_vocabulary(tokenized_tfidf)
        # print('Size of vocabulary after removing rare tokens: {}'.format(
        #     len(vocab)))
        tokenized_tfidf = []
        for i,row in enumerate(self.tokens.tokenized_pattern):
            tokenized_tfidf.append([token for token in row if
                                    token.lower() in Tokens.get_vocabulary(unique_tokenized)])
        pprint.pprint(tokenized_tfidf)
        return tokenized_tfidf


    def create_frequency_matrix(self, tokenized):
        frequency_matrix = []

        for tokens in tokenized:
            freq_table = {}
            for token in tokens:
                if token in freq_table:
                    freq_table[token] += 1
                else:
                    freq_table[token] = 1

            frequency_matrix.append(freq_table)

        return frequency_matrix


    def create_tf_matrix(self, freq_matrix):
        tf_matrix = []

        for f_table in freq_matrix:
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix.append(tf_table)

        return tf_matrix


    def create_documents_per_words(self, freq_matrix):
        word_per_doc_table = {}

        for f_table in freq_matrix:
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table


    def create_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = []

        for f_table in freq_matrix:
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix.append(idf_table)

        return idf_matrix


    def create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        tf_idf_matrix = []

        for f_table1, f_table2 in zip(tf_matrix, idf_matrix):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix.append(tf_idf_table)

        return tf_idf_matrix


    def remove_unnecessary(self, tokenized, tf_idf):
        tokenized_tfidf = []
        for i, row in enumerate(tokenized):
            print(row)
            weights = [tf_idf[i][token] for token in row]
            print(weights)
            top = np.mean(weights) + np.std(weights)
            print(top)
            tokenized_tfidf.append([v if weights[x] < top else '｟*｠' for x,v in enumerate(row)])
            print([v for x,v in enumerate(row) if weights[x] < top ])
        return tokenized_tfidf



        # f_matrix = self.create_frequency_matrix(tokenized)
        # tf_matrix = self.create_tf_matrix(f_matrix)
        # dpw = self.create_documents_per_words(tf_matrix)
        # idf_matrix = self.create_idf_matrix(tf_matrix, dpw, len(tokenized))
        # tf_idf = self.create_tf_idf_matrix(tf_matrix, idf_matrix)
        # tokenized_tfidf = self.remove_unnecessary(tokenized, tf_idf)
        #
        # print('Size of vocabulary after removing rare tokens: {}'.format(
        #     len(Tokens.get_vocabulary(tokenized_tfidf))))
        #
        # return tokenized_tfidf