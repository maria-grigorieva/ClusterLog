from clusterlogs import multistage
import pandas as pd
import pprint


filename = '../ddm.csv'
target = 'ddmerrordiag'
index = 'pandaid'

df = pd.read_csv(filename)
df.set_index(index, inplace=True)

result = multistage.exec(df, target, 'ddm.model')

pprint.pprint(result.results.to_dict('rows'))

pprint.pprint(result.in_cluster(0))