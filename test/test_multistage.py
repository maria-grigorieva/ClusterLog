from clusterlogs import multistage
import pandas as pd
import pprint


# filename = 'error_messages.csv'
# target = 'exeerrordiag'
# index = 'pandaid'
# model_name = 'exe.model'

filename = 'Project.csv'
target = 0
model_name = 'undrus.model'

df = pd.read_csv(filename, header=None)
df.dropna(inplace=True)

# df = pd.read_csv(filename)
# df.set_index(index, inplace=True)

result = multistage.exec(df, target, model_name)

pprint.pprint(result.out.patterns.to_dict('rows'))

#pprint.pprint(result.in_cluster(0))