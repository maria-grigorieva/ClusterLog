from clusterlogs import multistage
import pandas as pd
import pprint


filename = 'error_messages.csv'
target = 'exeerrordiag'
index = 'pandaid'

df = pd.read_csv(filename)
df.set_index(index, inplace=True)

result = multistage.exec(df, target, 'exe.model')

pprint.pprint(result.clusters.to_dict('rows'))

#pprint.pprint(result.in_cluster(0))