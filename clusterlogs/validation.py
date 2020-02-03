import editdistance
import numpy as np
import pandas as pd
from pyonmttok import Tokenizer

STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "mean_similarity",
              "std_similarity"]


CLUSTERING_ACCURACY = 0.8

class Output:

    def __init__(self):
        self.patterns = None
        self.tokenizer = Tokenizer("conservative", spacer_annotate=True)


    def cluster_statistics(self, item, row, messages):

        similarity = self.levenshtein_similarity(messages, 0)
        return {'cluster_name': item,
                'cluster_size': row['cluster_size'],
                'pattern': row['pattern'],
                'mean_similarity': np.mean(similarity),
                'std_similarity': np.std(similarity)}


    def statistics(self, df, target, groups):
        """
        :param clustered_df:
        :param output_mode: data frame
        :return:
        """

        patterns = []
        for index, row in groups.iterrows():
            messages = df.loc[row['indices'], target].values
            patterns.append(self.cluster_statistics(index, row, messages))
        return pd.DataFrame(patterns, columns=STATISTICS)\
                            .round(2)\
                            .sort_values(by='cluster_size', ascending=False)



    def levenshtein_similarity(self, rows, N):
        """
        :param rows:
        :return:
        """
        if len(rows) > 1:
            if N != 0:
                return (
                    [(1 - editdistance.eval(rows[0][:N], rows[i][:N]) / max(len(rows[0][:N]), len(rows[i][:N]))) for
                     i in
                     range(0, len(rows))])
            else:
                return (
                    [(1 - editdistance.eval(rows[0], rows[i]) / max(len(rows[0]), len(rows[i]))) for i
                     in
                     range(0, len(rows))])
        else:
            return 1
