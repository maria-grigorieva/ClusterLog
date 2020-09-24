import pandas as pd
import sys
import pprint
from clusterlogs import pipeline

def main():
    df = pd.read_csv('../samples/exeerror_1week.csv')
    df = df[['pandaid','exeerrordiag']]
    df.set_index('pandaid', inplace=True)
    target = 'exeerrordiag'

    # UPDATE
    # cluster = pipeline.Chain(df, target, mode='process', model_name='../models/exeerrors_01-01-20_05-20-20.model',matching_accuracy=0.8,
    #                          clustering_type='ML', output_file='../reports/exeerror_week_create.html', categorization=False,
    #                          generate_html_report=False)
    # CREATE
    cluster = pipeline.Chain(df, target, mode='create', model_name='../models/exeerrors_tmp.model',
                             matching_accuracy=0.8,
                             clustering_type='ML', output_file='../reports/exeerror_week_create.html')
    cluster.process()

    pprint.pprint(cluster.result)



if __name__ == "__main__":
    sys.exit(main())