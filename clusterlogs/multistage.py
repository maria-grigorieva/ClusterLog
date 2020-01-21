import sys
from .pipeline import ml_clustering

class exec():

    def __init__(self, df, target, model):

        self. df = df
        self.target = target
        self.model = model

        self.cluster = ml_clustering(df, target, mode='create', model_name=model)
        self.cluster.process()
        # all_df = cluster.patterns_stats
        self.main_df = self.cluster.patterns_stats[self.cluster.patterns_stats['cluster_size'] >= 100]
        outliers = self.cluster.patterns_stats[self.cluster.patterns_stats['cluster_size'] < 100]['cluster_name'].values
        self.outliers_df = self.cluster.df.iloc[outliers]

        self.cluster_outliers = ml_clustering(self.outliers_df, target, mode='process', model_name=model)
        self.cluster_outliers.process()
        self.outliers_df = self.cluster_outliers.patterns_stats

        self.common_df = self.main_df.append(self.outliers_df)

        self.common = ml_clustering(self.common_df, 'pattern', mode='update', model_name=model, finished=True)
        self.common.process()
