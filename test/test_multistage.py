from clusterlogs import multistage
import pandas as pd
import pprint


df = pd.read_csv('error_messages.csv', index_col=0)
df.set_index('pandaid', inplace=True)
target = 'exeerrordiag'

result = multistage.exec(df, target, 'word2vec.model')

pprint.pprint(result.common.patterns_stats.to_dict('rows'))

pprint.pprint(result.in_cluster(0))