from .pipeline import ml_clustering
from .cluster_output import Output
from time import time


class exec():

    def __init__(self, df, target, model, threshold=100):
        """
        :param df:
        :param target:
        :param model:
        :param threshold:
        """

        self.df = df
        self.target = target
        self.model = model
        self.threshold = threshold

        self.clusterization = ml_clustering(df, target, mode='create', model_name=model).process()

        all = self.clusterization.output.patterns
        self.clusterization.big, self.clusterization.small = self.split_clusters(all)

        if all.shape[0] > 200:
            self.result = self.reclusterization(all)
        else:
            self.result = all

        self.clusters, self.outliers = self.split_clusters(self.result)


    def in_cluster(self, label):
        df = self.result
        indices = df[df['cluster_name'] == str(label)]['indices'].values.tolist()[0]
        return self.df.loc[indices][self.target].values.tolist()


    def reclusterization(self, clusters):
        self.out = Output(self.df, self.target)
        self.out.postprocessing(clusters)
        return self.out.patterns


    def split_clusters(self, clusters):
        return clusters[clusters['cluster_size'] >= self.threshold], \
               clusters[clusters['cluster_size'] < self.threshold]


