import sys
from .pipeline import ml_clustering
from .cluster_output import Output
import pandas as pd
import pprint

class exec():

    def __init__(self, df, target, model, threshold=100):

        self.df = df
        self.target = target
        self.model = model
        self.threshold = threshold

        self._1 = ml_clustering(df, target, mode='create', model_name=model).process()

        all_clusters = self._1.patterns

        if all_clusters.shape[0] > 100:
            self.big_clusters = self._1.patterns[self._1.patterns['cluster_size'] >= self.threshold]
            outliers_indices = [i for sublist in self._1.patterns[self._1.patterns['cluster_size'] < self.threshold]['indices'].values for i in sublist]
            self.df_outliers = self.df.loc[outliers_indices]

            self._2 = ml_clustering(self.df_outliers, target, mode='process', model_name=model).process()
            self.small_clusters = self._2.patterns

            self.clusters = self.big_clusters.append(self.small_clusters, ignore_index = True, sort = False)

            self.outliers = self.clusters[self.clusters['cluster_size']<self.threshold]
            self.clusters.drop(self.outliers.index, axis=0, inplace=True)

        else:

            self.clusters = all_clusters


    def in_cluster(self, cluster_label):
        df = self.clusters
        indices = df[df['cluster_name'] == str(cluster_label)]['indices'].values.tolist()[0]
        return self.df.loc[indices][self.target].values.tolist()

