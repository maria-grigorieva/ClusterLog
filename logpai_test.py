import pandas as pd
# import sys
# import pprint
from clusterlogs.pipeline import Chain


def main():
    with open("../datasets/SSH.log") as f:
        big_data = [s[33:].lstrip("]: ") for s in f.readlines()]
    # train_file = open('../../datasets/train_SSH.txt', 'w')
    # for l in big_data:
    #     train_file.write(l)
    #
    # train_file = open('../../datasets/openssh.log', 'w')
    with open('../datasets/openssh.log') as f:
        data = [s[33:].lstrip("]: ") for s in f.readlines()]

    df = pd.DataFrame(data=data, columns=['message'])
    big_df = pd.DataFrame(data=big_data, columns=['message'])

    # big_df.to_csv('../samples/SSH.csv')
    # df.to_csv('../samples/SSH_2k.csv')
    # df = pd.read_csv('../samples/SSH_2k.log.txt')

    # CREATE MODE
    # with open("../samples/SSH_2k.log.txt") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])

    target = 'message'
    cluster = Chain(big_df, target, model_name='../models/SSH.model', mode='create',
                    clustering_type='ML', output_file='../reports/SSH_create.html')
    cluster.process()
    # pprint.pprint(cluster.result['pattern'].values)

    # UPDATE MODE
    # with open("../samples/SSH_2k.log.txt") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])
    # target = 'message'
    # pprint.pprint(df.head())

    cluster = Chain(df, target, model_name='../models/SSH.model', mode='process',
                    clustering_type='ML', output_file='../reports/SSH_process.html')
    cluster.process()

    # with open("../samples/train_SSH.txt") as f:
    #     data = f.readlines()
    # df = pd.DataFrame(data, columns=['message'])
    # target = 'message'
    # pprint.pprint(df.head())
    cluster = Chain(df, target, model_name='../models/SSH_small.model', mode='create',
                    clustering_type='ML', output_file='../reports/SSH_small.html')
    cluster.process()


if __name__ == "__main__":
    main()
