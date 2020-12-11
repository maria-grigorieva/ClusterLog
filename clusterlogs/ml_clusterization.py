import math
import numpy as np
import pandas as pd

from kneed import KneeLocator
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from .phraser import extract_common_phrases
from .Drain import LogParser
from .tokenization import get_vocabulary, detokenize_messages
from .data_preparation import clean_messages
from .sequence_matching import Match
import spacy
nlp = spacy.load("en_core_web_sm")

LIMIT = 30


class MLClustering:

    def __init__(self, df, groups, vectors, cpu_number, add_placeholder, method, tokenizer_type, pca):
        self.groups = groups
        self.df = df
        self.method = method
        self.vectors = vectors
        self.distances = None
        self.epsilon = None
        self.min_samples = 1 
        self.cpu_number = cpu_number
        self.add_placeholder = add_placeholder
        self.tokenizer_type = tokenizer_type
        self.diversity_factor = 0
        self.pca = pca


    def process(self):
        if self.method == 'dbscan':
            return self.dbscan()
        if self.method == 'hdbscan':
            return self.hdbscan()
        if self.method == 'hierarchical':
            return self.hierarchical()


    def dimensionality_reduction(self):
        n = self.vectors.detect_embedding_size(get_vocabulary(self.groups['sequence']))
        print('Number of dimensions is {}'.format(n))
        pca = PCA(n_components=n, svd_solver='full')
        pca.fit(self.vectors.sent2vec)
        return pca.transform(self.vectors.sent2vec)


    def kneighbors(self):
        """
        Calculates average distances for k-nearest neighbors
        :return:
        """
        k = round(math.sqrt(len(self.vectors.sent2vec)))
        print('K-neighbours = {}'.format(k))
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='auto').fit(self.vectors.sent2vec) #-1 means using all processors       
        distances, indices = nbrs.kneighbors(self.vectors.sent2vec)
        self.distances = [np.mean(d) for d in np.sort(distances, axis=0)]
       
    def epsilon_search(self):
        """
        Search epsilon for the DBSCAN clusterization
        :return:
        """
        kneedle = KneeLocator(self.distances, list(range(len(self.distances))), online=True)
        self.epsilon = np.mean(list(kneedle.all_elbows))
        if self.epsilon == 0.0:
            self.epsilon = np.mean(self.distances)


    def dbscan(self):
        """
        Execution of the DBSCAN clusterization algorithm.
        Returns cluster labels
        :return:
        """
        if self.pca:
            self.vectors.sent2vec = self.dimensionality_reduction()
        self.kneighbors()
        self.epsilon_search()
        self.cluster_labels = DBSCAN(eps=self.epsilon,
                                     min_samples=self.min_samples,
                                     n_jobs=self.cpu_number) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('DBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))
        return pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)


    def hdbscan(self):
        self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 else self.dimensionality_reduction()

        clusterer = HDBSCAN(min_cluster_size=10, min_samples=1)
        self.cluster_labels = clusterer.fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        print('HDBSCAN finished with {} clusters'.format(len(set(self.cluster_labels))))
        return pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)


    def hierarchical(self):
        """
        Agglomerative clusterization
        :return:
        """
        if len(self.vectors.sent2vec) >= 5000:
            self.vectors.sent2vec = self.vectors.sent2vec if self.vectors.w2v_size <= 10 \
                else self.dimensionality_reduction()
        self.cluster_labels = AgglomerativeClustering(n_clusters=25,
                                                      distance_threshold=None) \
            .fit_predict(self.vectors.sent2vec)
        self.groups['cluster'] = self.cluster_labels
        self.result = pd.DataFrame.from_dict(
            [item for item in self.groups.groupby('cluster').apply(func=self.gb_regroup)],
            orient='columns').sort_values(by=['cluster_size'], ascending=False)


    def gb_regroup(self, gb):
        m = Match(gb['tokenized_pattern'].values, add_placeholder=self.add_placeholder)
#         m2 = Match(gb['sequence'].values.tolist(), add_placeholder=self.add_placeholder)
        tokenized_pattern = []
#         tok_patt_cleaned=[]
        sequences = gb['tokenized_pattern'].values
#         sequences_cleaned=gb['sequence'].values.tolist()
        
        if len(sequences) > 1:
            m.matching_clusters(sequences, tokenized_pattern)
#             m2.matching_clusters(sequences_cleaned, tok_patt_cleaned)
        elif len(sequences) == 1:
            tokenized_pattern.append(sequences[0])
#             tok_patt_cleaned.append(sequences_cleaned[0])
        common_pattern = detokenize_messages(tokenized_pattern, self.tokenizer_type)
#         common_patt_cleaned = detokenize_messages(tok_patt_cleaned, self.tokenizer_type)#original in Maria 
        pattern_cl=[sublist[0] for sublist in gb['cleaned_strings'].values]
        text = '. '.join(clean_messages(common_pattern))
        #phrases_pyTextRank = Phraser(text, 'pyTextRank')
        # print('Extracting key phrases...')
        # phrases_RAKE = extract_common_phrases(text, 'rake_nltk')
        # Get all indices for the group
        indices = [i for sublist in gb['indices'].values for i in sublist]
        #code= [i for sublist in gb['error_code'].values for i in sublist]
        category = [i for sublist in gb['error_category'].values for i in sublist]
        #phase= [i for sublist in gb['failure_phase'].values for i in sublist]
        #scope = [i for sublist in gb['error_scope'].values for i in sublist]
        size = len(indices)
        phrases_RAKE = extract_common_phrases(text, 'RAKE')
        #phrases_RAKE = extract_common_phrases(text, 'rake_nltk')

        # doc = nlp(text)
        # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
        # for entity in doc.ents:
        #     print(entity.text, entity.label_)

        # return {'pattern': pattern,
        #         'indices': indices,
        #         'cluster_size': size,
        #         'common_phrases_RAKE': phrases_RAKE,
        #         'verbs': np.unique([token.lemma_ for token in doc if token.pos_ == "VERB"]).tolist(),
        #         'noun_phrases': np.unique([chunk.text for chunk in doc.noun_chunks]).tolist(),
        #         'entities': np.unique([entity.text for entity in doc.ents]).tolist()}

        return {'common_pattern':common_pattern, #message patterns made of only shared tokens (within the cluster)
#                 'common_patt_cleaned':common_patt_cleaned, #message patterns made of cleaned COMMON strings
                'cleaned_pattern': pattern_cl,#message patterns made of cleaned strings
                'indices': indices,
#                 'error_code': code,
                'error_category': category,
#                 'error_scope':scope,
#                 'failure_phase':phase,
                'cluster_size': size,
               'common_phrases_RAKE': phrases_RAKE}
