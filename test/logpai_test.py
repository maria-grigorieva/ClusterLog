import pandas as pd
import sys
import pprint
from clusterlogs import pipeline


def main():
    # with open("../samples/SSH.log") as f:
    #     big_data = [s[33:].lstrip("]: ") for s in f.readlines()]
    # train_file = open('../samples/train_SSH.txt', 'w')
    # for l in big_data:
    #     train_file.write(l)
    #
    # train_file = open('../samples/SSH_2k.log', 'w')
    # with open("../samples/SSH_2k.log.txt") as f:
    #     data = [s[33:].lstrip("]: ") for s in f.readlines()]
    #
    # df = pd.DataFrame(data=data, columns=['message'])
    # big_df = pd.DataFrame(data=big_data, columns=['message'])

    # big_df.to_csv('../samples/SSH.csv')
    # df.to_csv('../samples/SSH_2k.csv')
    # df = pd.read_csv('../samples/SSH_2k.log.txt')

    # CREATE MODE
    # with open("../samples/SSH_2k.log.txt") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])
    # target = 'message'
    # cluster = pipeline.Chain(df, target, model_name='../models/SSH_create.model', mode='create',
    #                          clustering_type='ML', output_file='../reports/SSH_create.html')
    # cluster.process()
    # pprint.pprint(cluster.result['pattern'].values)

    # UPDATE MODE
    # with open("../samples/SSH_2k.log.txt") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])
    # target = 'message'
    # pprint.pprint(df.head())
    # cluster = pipeline.Chain(df, target, model_name='../models/SSH.model', mode='update',
    #                 clustering_type='ML', output_file='../reports/SSH_update.html')
    # cluster.process()

    with open("../samples/train_SSH.txt") as f:
        data = f.readlines()
    df = pd.DataFrame(data, columns=['message'])
    target = 'message'
    pprint.pprint(df.head())
    cluster = pipeline.Chain(df.head(10000), target, model_name='../models/SSH_create.model', mode='create',
                             clustering_type='ML', output_file='../reports/SSH_create_10K.html')
    cluster.process()


if __name__ == "__main__":
    sys.exit(main())
