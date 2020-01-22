import sys
from .pipeline import ml_clustering
from .cluster_output import Output

class exec():

    def __init__(self, df, target, model):

        self.df = df
        self.target = target
        self.model = model

        self.cluster = ml_clustering(df, target, mode='create', model_name=model)
        self.cluster.process()

        all_df = self.cluster.patterns_stats


        if all_df.shape[0] > 40:
            self.main_df = self.cluster.patterns_stats[self.cluster.patterns_stats['cluster_size'] >= 100]
            outliers = self.cluster.patterns_stats[self.cluster.patterns_stats['cluster_size'] < 100]['cluster_name'].values
            self.outliers_df = self.cluster.df.iloc[outliers]

            self.cluster_outliers = ml_clustering(self.outliers_df, target, mode='process', model_name=model)
            self.cluster_outliers.process()
            self.outliers_df = self.cluster_outliers.patterns_stats

            self.common_df = self.main_df.append(self.outliers_df)

            self.results = self.reclusterization(self.common_df)

            # self.common = ml_clustering(self.common_df, 'pattern', mode='update', model_name=model, finished=True)
            # self.common.process()
            #
            # self.results = self.common.patterns_stats

        else:

            self.results = all_df


    def in_cluster(self, cluster_label):
        df = self.results
        indices = df[df['cluster_name'] == str(cluster_label)]['indices'].values.tolist()[0]
        return self.df.loc[indices][self.target].values.tolist()


    def reclusterization(self, stat_df):
        self.output = Output(self.df, self.target)
        self.patterns_stats = self.output.postprocessing(stat_df)
