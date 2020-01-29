import sys
from .pipeline import ml_clustering
from .cluster_output import Output
import pandas as pd
import pprint

class exec():

    def __init__(self, df, target, model, threshold=100):
        """
        Levels of clusterization:
        1) clusterize the initial dataframe
        2) clusterize outliers dataframe
        3) reclusterize patterns

        :param df:
        :param target:
        :param model:
        :param threshold:
        """

        self.df = df
        self.target = target
        self.model = model
        self.threshold = threshold

        self._1 = ml_clustering(df, target, mode='create', model_name=model).process()

        all_clusters = self._1.output.patterns
        self.big_clusters_1 = all_clusters[all_clusters['cluster_size'] >= self.threshold]
        self.small_clusters_1 = all_clusters[all_clusters['cluster_size'] < self.threshold]

        if self.small_clusters_1.shape[0] > 200:
            # outliers_indices = [i for sublist in all_clusters[all_clusters['cluster_size'] < self.threshold]['indices'].values for i in sublist]
            # self.df_outliers = self.df.loc[outliers_indices]
            #
            # self._2 = ml_clustering(self.df_outliers, target, mode='process', model_name=model).process()
            #
            # self.reclustered = self.big_clusters_1.append(self._2.output.patterns, ignore_index = True, sort = False)
            #
            # self.small_clusters_2 = self.reclustered[self.reclustered['cluster_size']<self.threshold]
            # self.big_clusters_2 = self.reclustered.drop(self.small_clusters_2.index, axis=0)

            self.out = Output(self.df, self.target)
            #self.out.postprocessing(self.reclustered)
            self.out.postprocessing(all_clusters)
            self.result = self.out.patterns
            self.outliers = self.result[self.result['cluster_size']<self.threshold]
            self.big_clusters = self.result.drop(self.outliers.index, axis=0)

        else:

            self.result = all_clusters
            self.outliers = self.result[self.result['cluster_size'] < self.threshold]
            self.big_clusters = self.result.drop(self.outliers.index, axis=0)


    def in_cluster(self, cluster_label):
        df = self.result
        indices = df[df['cluster_name'] == str(cluster_label)]['indices'].values.tolist()[0]
        return self.df.loc[indices][self.target].values.tolist()



